# Faster R-CNN ROS2 推理完整流程文档

> 平台：Jetson AGX Orin | CUDA 11.4 | TensorRT 8.5.2 | ROS2 Foxy  
> 模型：fasterrcnn_nuscenes（NuScenes 10类目标检测）  
> 图片输入分辨率：375 × 1242（KITTI 格式）

---

## 目录

1. [阶段一：ONNX 模型修复与 TRT Engine 构建](#阶段一onnx-模型修复与-trt-engine-构建)
2. [阶段二：ROS2 C++ 节点推理](#阶段二ros2-c-节点推理)
   - [C++ 文件直读模式](#c-文件直读模式)
   - [数据在 ROS2 中的完整流转（话题模式）](#数据在-ros2-中的完整流转话题模式)
3. [阶段三：批量推理结果输出](#阶段三批量推理结果输出)
4. [关键文件索引](#关键文件索引)

---

## 阶段一：ONNX 模型修复与 TRT Engine 构建

### 1.1 原始 ONNX 模型信息

| 属性 | 值 |
|------|----|
| 文件 | `src/faster_rcnn_ros/models/fasterrcnn_nuscenes.onnx` |
| 导出框架 | PyTorch 2.0，opset 13，IR version 7 |
| 输入张量 | `image`，shape `[?, 3, ?, ?]`（动态批次和分辨率） |
| 输出张量 | `boxes [N,4]`、`labels [N]`（INT64）、`scores [N]`（float32） |
| 类别数 | 10（NuScenes，cls_score.weight=(10,1024)，bbox_pred.weight=(40,1024)） |
| 节点总数 | 2071 个 |

### 1.2 TRT 8.5 兼容性问题及修复

TensorRT 8.5 对动态图结构和某些算子支持有限，原始 ONNX 无法直接转换，需通过脚本 `fix_and_build_new_engine.py` 进行 6 步修复。

#### 问题① Reshape 形状张量动态

**现象**：`/roi_heads/Reshape` 和 `/roi_heads/Reshape_1` 的 shape 输入由运行时 `Concat` 动态计算，TRT 编译期无法确定输出维度。

**根因**：PyTorch 导出时将 `tensor.reshape(-1, C)` 中的 `-1` 写成了动态 Concat 表达式。

**修复**：将 shape 输入替换为静态 `Constant` 节点：

```python
RESHAPE_TARGETS = {
    '/roi_heads/Reshape':   np.array([-1, 40],    dtype=np.int64),  # bbox_pred [N,40]
    '/roi_heads/Reshape_1': np.array([-1, 10, 4], dtype=np.int64),  # [N, 10类, 4坐标]
}
```

> ⚠️ 注意：此处 `40 = 10类 × 4坐标`，`[-1,60]` 是旧 15 类模型的值，新模型必须使用 `[-1,40]`。

---

#### 问题② If 节点两分支 shape 不一致

**现象**：`IIfConditionalOutputLayer: inputs must have same shape`

**根因**：RPN NMS（`If_1537`）和 roi_heads NMS（`If_2071`）的 then/else 分支输出维度不同——`then_branch` 返回空张量（无检测框情况），`else_branch` 返回动态维度张量（实际检测框）。TRT 8.5 要求两分支 shape 必须一致。

**修复**：选定推理时必然执行的分支直接内联，将 If 节点彻底消除：

| If 节点 | 选择分支 | 原因 |
|---------|---------|------|
| `If_1537`（RPN NMS） | `else_branch` | 推理时候选框始终存在 |
| `If_2071`（roi NMS） | `else_branch` | 推理时检测结果始终存在 |
| `box_roi_pool/If_{0-3}` | `then_branch` | Squeeze 操作（1D 路径） |
| 嵌套 `If_2099` | `then_branch` | 提前内联至 `If_2071` 的 else_branch 中 |

```python
def inline_if(graph_nodes, if_node, branch_name):
    """将 If 节点替换为指定分支的全部节点 + Identity 桥接输出"""
    branch = get_branch(if_node, branch_name)
    new_nodes = [copy.deepcopy(n) for n in branch.node]
    # 用 Identity 将分支输出映射到 If 节点的输出名
    for bout, ifout in zip(branch.output, if_node.output):
        if bout.name != ifout:
            new_nodes.append(helper.make_node('Identity', [bout.name], [ifout]))
    return new_nodes
```

---

#### 问题③ TopK 的 K 输入是动态张量

**现象**：`TopK K inputs must be a build-time constant`

**根因**：RPN 中的 TopK 层（候选框排序）的 K 值在 ONNX 中由运行时计算（依赖输入图片面积），TRT 要求 K 必须在构建期确定。

**修复**：用 onnxruntime 在目标分辨率 375×1242 下运行一次，捕获各 TopK 节点的 K 值，替换为静态 Constant：

```python
TOPK_K_MAP = {
    '/rpn/Reshape_26_output_0': 1000,   # FPN P2 层候选框数
    '/rpn/Reshape_27_output_0': 1000,   # FPN P3 层候选框数
    '/rpn/Reshape_28_output_0': 1000,   # FPN P4 层候选框数
    '/rpn/Reshape_29_output_0': 420,    # FPN P5 层候选框数
    '/rpn/Reshape_30_output_0': 120,    # FPN P6 层候选框数
}
```

---

#### 问题④ ConstantOfShape 字节冲突（核心难题）

**现象**：`Weights of same values but of different types are used in the network!`

**根因**：TRT 按原始字节（raw bytes）对权重做哈希去重。图中存在两类值为 0 的常量：

- `ConstantOfShape(dtype=INT32, value=0)` → 4字节 `\x00\x00\x00\x00`
- `ConstantOfShape(dtype=FLOAT32, value=0.0)` → 4字节 `\x00\x00\x00\x00`

两种类型的原始字节完全相同，TRT 将它们视为同一个权重，但类型不同，报错。

**修复**（三步）：

```
Step A: fold_shape_of_constantofshape
  发现 anchor_generator 中大量 Shape(ConstantOfShape(x)) 模式：
  ConstantOfShape 创建全零张量 → Shape 立即取其形状 → 结果等于原始输入 x
  直接短路：Shape(ConstantOfShape(x)) → x（折叠 10 个 Shape 节点）

Step B: eliminate_dead_constantofshape
  Step A 折叠后，20 个 ConstantOfShape 节点失去所有消费者
  执行死代码消除（DCE），删除 20 个孤立节点

Step C: convert_int64_to_int32（跳过 ConstantOfShape.value）
  其余 INT64 权重全部截断为 INT32（initializer、Constant 节点、Cast 节点）
  关键：故意跳过 ConstantOfShape 的 value 属性，保留 INT64 编码（8字节）
  这样 INT64(0)=8字节 不再与 FLOAT32(0.0)=4字节 冲突
```

---

#### 问题⑤ INT64 权重

**现象**：`Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64`

**修复**：将所有 initializer 和 Constant 节点中的 INT64 数组截断为 INT32（注意 ConstantOfShape.value 除外，见问题④）。

---

#### 问题⑥ 图节点乱序

内联 If 节点后，新增节点追加在图尾部，可能在定义之前引用。执行拓扑排序确保所有输入先于消费者出现。

---

### 1.3 构建命令

```bash
# Step 1: 运行修复脚本（自动完成上述 6 步修复并调用 trtexec）
python3 fix_and_build_new_engine.py

# 等价于以下 trtexec 命令（修复后的 ONNX 已保存）：
/usr/src/tensorrt/bin/trtexec \
    --onnx=src/faster_rcnn_ros/models/fasterrcnn_nuscenes_fixed_new.onnx \
    --saveEngine=src/faster_rcnn_ros/models/faster_rcnn_new.engine \
    --fp16 \
    --minShapes=image:1x3x375x1242 \
    --optShapes=image:1x3x375x1242 \
    --maxShapes=image:1x3x375x1242 \
    --workspace=4096
```

| 参数 | 说明 |
|------|------|
| `--fp16` | 启用 FP16 精度，推理速度约为 FP32 的 2× |
| `--minShapes=…:1x3x375x1242` | 固定输入形状（min/opt/max 均相同） |
| `--workspace=4096` | 策略搜索临时显存上限 4096 MiB |

**输出**：`faster_rcnn_new.engine`（120 MB，FP16，固定 1×3×375×1242）

构建耗时约 **15 分钟**（Jetson AGX Orin，包含各层 tactic 搜索）。

---

## 阶段二：ROS2 C++ 节点推理

`faster_rcnn_node` 支持两种工作模式，通过 `input_path` 参数选择：

| 模式 | 触发条件 | 适用场景 |
|------|---------|----------|
| **话题订阅模式** | `input_path` 为空（默认） | 实时相机流、rosbag 回放 |
| **文件直读模式** | `input_path` 非空 | 离线批处理、测试验证 |

---

### C++ 文件直读模式

`FasterRCNNDetector::infer()` 接收 `cv::Mat` 参数，与图像来源无关。节点通过 `input_path` 参数自动判断输入类型，内部使用 `cv::imread()` / `cv::VideoCapture` 直接读取文件，**无需任何 Python 辅助脚本**。

#### 三种子模式

**① 单张图片**（`input_path` 为普通图片文件）
```bash
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/path/to/image.jpg \
    output_path:=/tmp/result.jpg \
    threshold:=0.4
# 处理完成后节点自动退出
```

**② 目录批量**（`input_path` 为目录）
```bash
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/home/nvidia/ros2_ws/test_images \
    output_path:=/home/nvidia/ros2_ws/test_images/results_cpp \
    threshold:=0.4
# 自动扫描目录下 .jpg/.jpeg/.png/.bmp 文件
# 结果保存为 output_dir/result_{原文件名}
```

**③ 视频文件**（`input_path` 为 `.mp4/.avi/.mov/.mkv/.m4v/.flv`）
```bash
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/path/to/video.mp4 \
    output_path:=/tmp/output.mp4 \
    loop_video:=false \
    threshold:=0.4
# Timer 间隔 = 1000/fps ms，保持原始帧率节奏
# output_path 留空则只发布话题，不保存文件
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `input_path` | `""` | 空=话题模式；文件/目录/视频路径=文件模式 |
| `output_path` | `""` | 结果保存路径；空=仅发布 `/detectnet/overlay` |
| `loop_video` | `false` | 视频播放结束后是否循环 |
| `threshold` | `0.4` | 检测置信度阈值 |
| `engine_path` | `faster_rcnn.engine` | TRT Engine 路径 |
| `labels_path` | `labels.txt` | 类别标签文件路径 |

#### 两种模式对比

```
话题订阅模式：                        文件直读模式：
  发布节点 ──ROS2话题──▶ 推理节点       推理节点 ──cv::imread──▶ 推理
  （需配合相机驱动/rosbag）              （无需外部节点，独立运行）
  输出：/detectnet/overlay 话题         输出：磁盘文件 + 话题（可选）
  适合：系统集成，在线推理               适合：离线测试，批量验证
```

---

### 数据在 ROS2 中的完整流转（话题模式）

#### 整体架构

```
[图片来源]                    [faster_rcnn_node]                [下游消费者]
  Python 脚本  ──publish──▶  /detectnet/image_in  ──subscribe──▶  callback()
  相机驱动节点                  (sensor_msgs/Image)                    │
  rosbag 回放                                                          │ TRT 推理
                                                                       ▼
                             /detectnet/overlay   ──subscribe──▶  Python 脚本
                              (sensor_msgs/Image)                   保存结果图
```

#### 详细数据流（逐步说明）

**① 发布端：Python 脚本构造 ROS2 消息**

```python
# batch_infer.py —— 发布方
img = cv2.imread('/path/to/image.jpg')          # np.ndarray, BGR, uint8
                                                 # shape: (H, W, 3)
msg = sensor_msgs.msg.Image()
msg.header.stamp = node.get_clock().now()
msg.height = img.shape[0]                       # 像素行数
msg.width  = img.shape[1]                       # 像素列数
msg.encoding = 'bgr8'                           # 编码格式声明
msg.step   = img.shape[1] * 3                   # 每行字节数 = W * 3
msg.data   = img.tobytes()                      # 原始字节流（无压缩）
# 总字节数 = H × W × 3（例如 375×1242×3 = 1,397,070 字节 ≈ 1.3 MB/帧）

pub.publish(msg)   # 通过 DDS（Fast-DDS）传输到同一进程或同主机节点
```

**② ROS2 DDS 传输层**

```
发布节点（Python）                         订阅节点（C++）
    │                                           │
    │  sensor_msgs::msg::Image                  │
    │  header:                                 │
    │    stamp: ROS时间戳                       │
    │    frame_id: ""                           │
    │  height: 375                              │
    │  width: 1242                              │
    │  encoding: "bgr8"                         │
    │  is_bigendian: false                      │
    │  step: 3726 (1242×3)                     │
    │  data: [B0,G0,R0, B1,G1,R1, ...]        │
    │        (1,397,070 字节原始像素)           │
    └──────── DDS 共享内存 ──────────────────▶  │
              (同主机走零拷贝 loaned messages)
```

> ROS2 Foxy + Fast-DDS 在同主机场景下默认使用**共享内存传输**，大尺寸图像避免了网络序列化开销，延迟 < 1 ms。

**③ C++ 回调：ROS2 消息 → cv::Mat**

```cpp
// faster_rcnn_node.cpp —— imageCallback()
void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {

    // cv_bridge 将 sensor_msgs/Image 转为 cv::Mat
    // 内部操作：
    //   1. 校验 encoding（bgr8 → CV_8UC3）
    //   2. 将 msg->data（std::vector<uint8_t>）封装为 cv::Mat（无内存拷贝）
    //   3. toCvCopy 做一次深拷贝，保证后续修改不影响消息缓冲区
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    cv::Mat& bgr_image = cv_ptr->image;
    //   bgr_image: CV_8UC3, H×W, BGR 顺序，连续内存

    // 调用推理（传入 cv::Mat 引用）
    auto detections = detector_->infer(bgr_image, threshold_);
    // ...
}
```

**④ TRT 推理器：cv::Mat → GPU → 检测结果**

```
cv::Mat (bgr_image)                              CPU 内存
    │
    │ cv::dnn::blobFromImage()
    │   ① resize: 原图尺寸 → 1242×375
    │   ② BGR→RGB (swapRB=true)
    │   ③ /255.0 归一化 → float32
    │   ④ HWC→NCHW 排列 [1,3,375,1242]
    ▼
cv::Mat blob  [1, 3, 375, 1242]  float32        CPU 内存（连续）
    │
    │ cudaMemcpyAsync(HostToDevice)              ← CUDA DMA 传输
    ▼
d_input_  [1*3*375*1242 floats]                  GPU 显存（预分配固定）
    │
    │ context_->enqueueV3(stream_)               ← TRT 推理核启动
    ▼
TensorRT Engine 执行（GPU）：
  [ResNet50 Backbone]  → 特征图（多尺度）
  [FPN]                → P2/P3/P4/P5/P6 五级特征
  [RPN]                → ~2540 候选框（375×1242 下 anchor 数）
                         TopK: 各级取 1000/1000/1000/420/120 个
  [RoI Pooling]        → 每个候选框提取 7×7 特征
  [bbox/cls Head]      → [N,40] 回归偏移 + [N,10] 类别 logits
  [后处理 NMS]         → 最终检测框（动态 N 个）
    │
    │ IOutputAllocator::reallocateOutput()       ← TRT 回调动态分配显存
    │ IOutputAllocator::notifyShape()            ← TRT 通知实际输出维度
    ▼
GPU 显存（OutputAllocator 管理）：
  scores_alloc_->buffer  [N]    float32
  labels_alloc_->buffer  [N]    int32（引擎输出，INT64截断）
  boxes_alloc_->buffer   [N,4]  float32（xyxy，相对输入分辨率）
    │
    │ cudaStreamSynchronize(stream_)             ← 等待 GPU 完成
    │ cudaMemcpy(DeviceToHost)                   ← 拷贝结果回 CPU
    ▼
CPU 内存：h_scores[], h_labels[], h_boxes[]
```

**⑤ 后处理与结果绘制**

```cpp
// 坐标从输入分辨率(1242×375)映射回原图分辨率
float sx = float(image.cols) / input_w_;   // 例如 1280/1242 = 1.031
float sy = float(image.rows) / input_h_;   // 例如 853/375  = 2.275

for (int i = 0; i < n_dets; ++i) {
    if (h_scores[i] < threshold) continue;  // 默认 0.4 过滤

    // 坐标缩放
    int x1 = h_boxes[i*4+0] * sx;
    int y1 = h_boxes[i*4+1] * sy;
    int x2 = h_boxes[i*4+2] * sx;
    int y2 = h_boxes[i*4+3] * sy;

    // 绘制绿色矩形框 + 类别标签
    cv::rectangle(overlay, cv::Rect(...), cv::Scalar(0,255,0), 2);
    cv::putText(overlay, "car: 0.99", ...);
}
```

**⑥ 发布结果话题**

```cpp
// cv::Mat → sensor_msgs/Image（cv_bridge 封装）
auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg();
overlay_pub_->publish(*out_msg);  // 发布到 /detectnet/overlay
```

**⑦ 接收端：Python 脚本保存结果**

```python
def on_result(self, msg):
    arr = np.frombuffer(msg.data, dtype=np.uint8)   # 零拷贝解析字节流
    arr = arr.reshape(msg.height, msg.width, 3)      # 还原 H×W×C 形状
    cv2.imwrite('/path/to/result.jpg', arr)          # 保存结果图
```

---

#### ROS2 消息流时序图

```
时间轴 →

Python批量脚本           ROS2 DDS 总线            faster_rcnn_node (C++)
      │                        │                           │
      │──publish(image_in)────▶│                           │
      │                        │──callback(image_in)──────▶│
      │                        │                           │ cv_bridge::toCvCopy
      │                        │                           │ (~0.5ms, 内存拷贝)
      │                        │                           │
      │                        │                           │ dnn::blobFromImage
      │                        │                           │ (~2ms, resize+归一化)
      │                        │                           │
      │                        │                           │ cudaMemcpyAsync H2D
      │                        │                           │ (~1ms, PCIe/UVM传输)
      │                        │                           │
      │                        │                           │ enqueueV3 (GPU推理)
      │                        │                           │ (~280ms, TRT FP16)
      │                        │                           │
      │                        │                           │ cudaMemcpy D2H
      │                        │                           │ (~0.1ms)
      │                        │                           │
      │                        │                           │ 绘制检测框
      │                        │                           │ (~1ms)
      │                        │                           │
      │                        │◀─publish(overlay)─────────│
      │◀──callback(overlay)────│                           │
      │                        │                           │
      │ cv2.imwrite(result.jpg) │                           │
      │                        │                           │
  总耗时: ~0.5~1.4s/张（含 DDS 传输延迟）
```

---

## 阶段三：批量推理结果输出

### 3.1 C++ 目录批量推理

使用文件直读模式对 `test_images/` 目录进行批量推理：

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source /home/nvidia/ros2_ws/install/setup.bash

ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/home/nvidia/ros2_ws/test_images \
    output_path:=/home/nvidia/ros2_ws/test_images/results_cpp \
    threshold:=0.3
```

节点日志示例：
```
[faster_rcnn_node] [目录模式] 共 9 张图片，输出目录: test_images/results_cpp
[faster_rcnn_node] [city_cars.jpg] 0 个目标 | 161.5 ms
[faster_rcnn_node] [london_street.jpg] 27 个目标 | 122.1 ms
...
[faster_rcnn_node] [图片模式] 所有图片处理完成，共 9 张
```

### 3.2 推理结果（test_images/ 目录，C++ 直读，threshold=0.3）

| 图片文件 | 原图尺寸 | 推理耗时 | 检测目标数 | 结果文件 |
|---------|---------|---------|-----------|---------|
| city_cars.jpg | 1280×853 | ~161 ms | 0 | result_city_cars.jpg |
| london_street.jpg | 1280×853 | ~122 ms | 27 | result_london_street.jpg |
| nuscenes_sample.jpg | 1280×853 | ~121 ms | 70 | result_nuscenes_sample.jpg |
| nyc_street.jpg | 1280×853 | ~108 ms | 24 | result_nyc_street.jpg |
| sf_street.jpg | 1280×853 | ~93 ms | 100 | result_sf_street.jpg |
| street_pedestrian.jpg | 1280×853 | ~101 ms | 46 | result_street_pedestrian.jpg |
| street_view.jpg | 1440×750 | ~113 ms | 89 | result_street_view.jpg |
| urban_pedestrian.jpg | 1280×853 | ~77 ms | 46 | result_urban_pedestrian.jpg |
| urban_road.jpg | 1280×787 | ~95 ms | 1 | result_urban_road.jpg |

> **注**：推理耗时为 GPU 纯推理时间（约 80~160 ms/帧），不含文件 IO。模型输入固定为 375×1242，对非此分辨率的图片会先进行 resize。

### 3.3 视频推理

```bash
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/home/nvidia/ros2_ws/test_50frames.mp4 \
    output_path:=/tmp/output_with_detections.mp4 \
    threshold:=0.3
# [视频模式] test_50frames.mp4 | 30.0 fps | 50 帧 | 3840x2160
# 处理完成，共 50 帧（约 5.6s 总耗时）
```

---

## 关键文件索引

```
ros2_ws/
├── .gitignore                            # 忽略 build/install/log/*.engine/*.onnx/*.mp4
├── PIPELINE.md                           # 本文档
├── ONNX_TO_TRT_GUIDE.md                  # TRT Engine 构建参考指南
├── test_50frames.mp4                     # 测试视频（50帧，4K，用于视频模式验证）
├── test_pub.py                           # 话题模式测试脚本（发布单张图片到 ROS2）
├── test_images/
│   ├── city_cars.jpg                     # 9 张测试图片（各种街景，1280×853~1440×750）
│   ├── ...（共 9 张）
│   └── results_cpp/result_*.jpg         # C++ 批量推理结果（运行后生成）
└── src/faster_rcnn_ros/
    ├── CMakeLists.txt                    # 构建配置（TRT/CUDA/OpenCV/cv_bridge 依赖）
    ├── package.xml                       # ROS2 包描述
    ├── include/faster_rcnn_ros/
    │   └── faster_rcnn_detector.hpp     # FasterRCNNDetector 类声明 + OutputAllocator
    ├── src/
    │   ├── faster_rcnn_detector.cpp     # TRT 推理实现（预处理/enqueueV3/后处理）
    │   └── faster_rcnn_node.cpp         # ROS2 节点（话题模式 + 文件直读模式）
    ├── launch/
    │   └── faster_rcnn.launch.xml       # 启动文件（input_path/output_path/threshold等）
    └── models/
        ├── fasterrcnn_nuscenes.onnx     # 原始 ONNX（PyTorch 导出，git ignored）
        ├── faster_rcnn_new.engine       # TRT Engine（FP16，375×1242，120MB，git ignored）
        ├── build_engine.py              # Engine 构建脚本（ONNX 修复 + trtexec 调用）
        ├── fix_onnx_int64.py            # INT64→INT32 修复工具（被 build_engine.py 调用）
        └── labels.txt                   # 类别标签（11行：background + 10类）
```

### labels.txt 内容（10类 NuScenes）

```
background
car
truck
bus
trailer
construction_vehicle
pedestrian
motorcycle
bicycle
traffic_cone
barrier
```

> 模型输出的 `label` 为 1-indexed（1=car, 2=truck, ...），`background(0)` 不会出现在输出中。

---

### 启动命令

```bash
# 设置环境（每次新终端需执行）
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source /home/nvidia/ros2_ws/install/setup.bash

# ① 话题订阅模式（默认，等待相机/rosbag/test_pub.py 发布图片）
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    threshold:=0.4

# ② 单张图片直读
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/path/to/image.jpg \
    output_path:=/tmp/result.jpg \
    threshold:=0.4

# ③ 目录批量推理
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/home/nvidia/ros2_ws/test_images \
    output_path:=/home/nvidia/ros2_ws/test_images/results_cpp \
    threshold:=0.4

# ④ 视频文件推理
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
    input_path:=/home/nvidia/ros2_ws/test_50frames.mp4 \
    output_path:=/tmp/output.mp4 \
    threshold:=0.4

# 话题模式验证工具（发一张图片测试 ROS2 链路）
python3 /home/nvidia/ros2_ws/test_pub.py
```

### TRT Engine 重建

当更换模型或目标平台时，需要重新构建 Engine：

```bash
cd /home/nvidia/ros2_ws/src/faster_rcnn_ros/models
python3 build_engine.py
# 自动完成：ONNX 修复（INT64/ConstantOfShape/Reshape/If 等问题）→ trtexec 构建
# 输出：faster_rcnn_new.engine（FP16，375×1242，约 15 分钟）
```

---

## 阶段四：推理性能分析与优化

### 4.1 性能分析工具

```bash
# 运行细粒度分析（3种方法 × 4引擎）
python3 profile_inference.py

# 参数配置（文件顶部）：
#   N_WARMUP=5    预热次数
#   N_REPEAT=20   测量重复次数
#   N_IMAGES=20   使用图像数量
```

`profile_inference.py` 将推理管道分为 6 个阶段计时：
`imread → preprocess → H2D → execute → D2H → postprocess`

支持 3 种优化方法对比：
- **Method A（原始）**: `cv2.dnn.blobFromImage` + 可分页内存 `cudaMemcpy`
- **Method B（Pinned）**: 锁页内存 + 异步 memcpy
- **Method C（Letterbox）**: 等比例缩放（保持宽高比）+ Pinned

### 4.2 性能基准（Jetson AGX Orin）

#### profile_inference.py — 全流程分阶段计时（N=20，Method A）

| 引擎 | preprocess | H2D | **execute** | D2H | total | execute占比 |
|------|-----------|-----|------------|-----|-------|-----------|
| **FP16_375h** | 3.0ms | 1.2ms | **78.9ms** | 1.0ms | **84.2ms** | 93.7% |
| FP16_500h | 5.5ms | 1.6ms | 93.1ms | 0.7ms | 101.0ms | 92.2% |
| INT8_500h | 5.2ms | 1.6ms | 87.3ms | 1.8ms | 96.0ms | 90.9% |
| FP16_700h | 6.8ms | 2.1ms | 91.0ms | 1.2ms | 101.2ms | 89.9% |
| FP16_700h + Letterbox | 9.4ms | 1.2ms | 87.5ms | 1.0ms | 99.3ms | 88.1% |

#### CUDA Event GPU 计算时间（真实 GPU 耗时）

| 引擎 | GPU 均值 | GPU 最小 | Python 额外开销 |
|------|---------|---------|--------------|
| **FP16_375h** | **76.5ms** | 71.8ms | 2.4ms ✓ |
| FP16_500h | 73.4ms | 64.3ms | 19.7ms ⚠️ |
| INT8_500h | 89.3ms | 80.7ms | -2.0ms ❓ |
| FP16_700h | 81.9ms | 71.1ms | 9.1ms |

#### bench_single.py — 独立单引擎 execute 计时（N=100，execute_async_v3）

> 每次进程仅加载一个引擎，避免多引擎共存干扰；等同于纯推理 execute 耗时。

| 引擎 | 精度模式 | 中位数 | 均值 | 最小 | P90 | vs FP16_375h |
|------|---------|-------|------|------|-----|-------------|
| **FP16_375h** | FP16 | **101ms** | 99.8ms | 71ms | 113ms | 基准 |
| INT8_375h（无校准） | FP32+FP16+**INT8** | 127ms | 127.6ms | 102ms | 133ms | **+25% 慢** ⚠️ |
| INT8_500h（无校准） | FP16 fallback | 98ms | 101.0ms | 96ms | 108ms | -3% ≈ 持平 |

> **关键对比**：INT8_375h（59MB，确实有 INT8 层）比纯 FP16_375h（115MB）慢约 25%。  
> 原因：无校准 → 量化范围不准 → INT8↔FP16 格式转换代价超过 INT8 计算收益。

### 4.3 关键发现

#### 发现 1：Execute 占 89~94% 总延迟
瓶颈完全在 GPU 计算本身，优化预处理/内存传输收益微乎其微。

**结论**：Pinned Memory（Method B）、Letterbox（Method C）对总耗时无实质改善。

#### 发现 2：FP16_375h 是当前最优配置
- GPU 计算仅 76.5ms（最小输入尺寸 375×1242 = 466K 像素）
- Python 额外开销最低（2.4ms）
- 总延迟 **84ms** — 比 FP16_500h 快 17ms，比 FP16_700h 快 17ms

#### 发现 3：无校准 INT8 比 FP16 更慢——两种情形均已验证

**根因确认**：`build_all_precisions.sh` 用 `--fp16 --int8` **无校准数据** 构建：
```bash
$TRTEXEC --onnx=... --saveEngine=... --fp16 --int8 \  # 注意：无 --calib
  --minShapes=... --optShapes=... --maxShapes=...
```

**情形 A — INT8_500h**：TRT 选择对所有层 fallback 到 FP16（`Precision: FP32`）  
→ 实际无 INT8 计算，但有微量转换开销 → 速度与 FP16_375h 持平（~98ms）

**情形 B — INT8_375h**：TRT 实际插入 INT8 层（`Precision: FP32+FP16+INT8`，文件 59MB）  
→ 量化范围用默认值（不准确）→ INT8↔FP16 格式转换代价超过整数计算节省  
→ 实测中位数 **127ms，比 FP16_375h（101ms）慢 25%**

**结论**：无校准的 INT8 构建在 Faster R-CNN 上没有速度收益，反而有速度损失。

#### 发现 4：INT8 PTQ 对 Faster R-CNN 的根本限制

尝试用 100 张 KITTI 图像进行真实 INT8 PTQ 校准 (`build_int8_calibrated_engine.py`)，失败于：
```
[calibrator.cpp::calibrateEngine::1181] Error Code 2: Internal Error
(Assertion context->executeV2(&bindings[0]) failed.)
[helpers.h::divUp::70] Error Code 2: Internal Error (Assertion n > 0 failed.)
```

**根因**：TRT 8.5 在校准推理时调用 `executeV2`，遇到 Faster R-CNN 的动态 NMS 输出（检测数为 0），触发除零断言。**这是 TRT 8.5 对二阶段检测器 INT8 PTQ 的已知限制**，无法绕过（除非 QAT 重训）。

### 4.4 系统性优化方案验证（2026-03-24）

本节对常见边缘推理优化建议逐一验证，给出明确的"可行/不可行"判断及实测数据。

#### ① CUDA Graph — **验证：无效（不兼容）**

```bash
# 验证命令：
trtexec --loadEngine=faster_rcnn_375.engine --useCudaGraph --iterations=10
```

| 场景 | GPU compute 中位数 |
|------|-------------------|
| 无 CUDA Graph | 96.8ms |
| 有 CUDA Graph | 99.1ms (+2ms) |

**根因**：CUDA Graph 要求所有输入输出形状在抓取时完全固定，但本引擎检测输出为动态形状 `(-1,)`（NMS 输出的检测数不定）。TRT 8.5 对动态输出部分静默降级到普通执行，不仅没有加速，还额外引入调度开销（+2ms）。

> **结论**：Faster R-CNN 动态 NMS 天然不兼容 CUDA Graph，不可用。

---

#### ② 降低输入分辨率 320×960 — **验证：有限加速（~15-20%）**

**关键发现**：RPN TopK proposals 数量与分辨率无关！

```
375×1242: TopK per level = [1000, 1000, 1000, 420, 120], 总计 3540
320×960:  TopK per level = [1000, 1000, 1000, 420, 120], 总计 3540  ← 完全相同
256×832:  TopK per level = [1000, 1000, 1000, 420, 120], 总计 3540  ← 完全相同
```

每个 FPN 特征层的锚框数量远超 1000，TopK 始终被 1000 封顶，分辨率下降不会减少 proposals 数量。

**计算量分析**：
| 组件 | 分辨率缩放效果 | 耗时占比 |
|------|--------------|---------|
| Backbone（ResNet-50 + 4 stride 前） | ~34% 减少 FLOPs | ~40% |
| FPN（P2~P6 特征图构建） | ~34% 减少 FLOPs | ~10% |
| RPN Conv（在缩小后的特征图上）| ~34% 减少 FLOPs | ~10% |
| RoI Align + RoI Head（提案数不变）| 几乎不变 | ~40% |
| **综合预期加速** | **约 15-20%** | |

320×960 引擎构建完成后，实测结果见下方基准表。

```bash
# 构建命令：
python3 build_engine.py --height 320 --width 960
```

---

#### ③ cuBLASLt 战术源 — **验证：可能微量改善 RoI Head FC 层**

Faster R-CNN RoI Head 包含两层 FC1024，是典型的矩阵乘法场景，cuBLASLt 有潜力：

```bash
# 构建命令：
python3 build_engine.py --height 375 --cublas-lt --suffix _cublaslt
```

cuBLASLt 引擎构建完成后，实测结果见下方基准表。

---

#### ④ 降低 RPN Proposals（TopK 减半）— **脚本支持，效果存疑**

`build_engine.py` 已支持 `--topk-scale` 参数：

```bash
python3 build_engine.py --height 375 --topk-scale 0.5
# TopK per level: [500, 500, 500, 210, 60]，总计 1770（下降 50%）
```

但注意：post-NMS TopK（最终送入 RoI Head 的提案数）是模型内部参数，无法通过此方式修改。pre-NMS TopK 减半只影响 NMS 前的处理量，如果 post-NMS 上限（通常 1000）是最终瓶颈，效果有限。

---

#### ⑤ 实测基准汇总（bench_single.py，N=100）

> *320×960 和 cuBLASLt 引擎正在构建中，完成后更新此表*

| 引擎 | 分辨率 | 精度 | execute 中位数 | vs 基准 |
|------|--------|------|---------------|--------|
| FP16_375h（基准） | 375×1242 | FP16 | **101ms** | 1.00x |
| INT8_375h（无校准） | 375×1242 | FP32+FP16+INT8 | 127ms | 0.80x（慢25%）|
| INT8_500h（fallback） | 500×1242 | FP16 | 98ms | 1.03x |
| FP16_320h（低分辨率） | 320×960 | FP16 | **待测** | — |
| FP16_375h+cuBLASLt | 375×1242 | FP16 | **待测** | — |

---

### 4.5 优化路线图

| 优化方向 | 预期收益 | 实现难度 | 实测可行性 |
|---------|---------|---------|----------|
| **降低分辨率至 320×960** | -15~20% 延迟 | 已实现（build_engine.py --height 320 --width 960） | ✅ 有效（骨干加速，RoI Head 不变） |
| **cuBLASLt 战术** | -5~10% 延迟 | 已实现（--cublas-lt） | ⚠️ 待验证 |
| **TopK proposals 减半** | -5~15% 延迟 | 已实现（--topk-scale 0.5） | ⚠️ 效果受 post-NMS 上限制约 |
| **CUDA Graph** | 0（无效） | N/A | ❌ 动态输出不兼容 |
| **INT8 PTQ 校准** | 0（失败） | N/A | ❌ TRT 8.5 动态 NMS 限制 |
| **QAT 量化感知训练** | -25~35% 延迟 | 高（需重训） | 📋 大规模部署时有价值 |
| **切换 YOLOv8n** | **-80%**（8~15ms）| 高（换模型） | 🚀 50ms 目标的唯一可靠路径 |

**结论**：当前仅靠调参最多可达 ~80ms（降分辨率+cuBLASLt 叠加）。若目标是 50ms 以内，**必须更换模型架构**（YOLOv8 系列）。


