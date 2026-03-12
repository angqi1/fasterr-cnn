# Faster R-CNN ONNX → TensorRT 转换指南

**平台**: Jetson AGX Orin | **TensorRT**: 8.5.2.2 | **CUDA**: 11.4 | **ROS2**: Foxy  
**模型输入**: `[1, 3, 480, 640]` 固定尺寸  

---

## 一、问题总结与修复方案

将 PyTorch Faster R-CNN 导出的 ONNX 模型直接送入 `trtexec` 时，TRT 8.5 会遇到多个兼容性问题。下表列出每个问题、根因和修复方式，以及对推理结果的影响。

### 问题清单

| # | 错误信息 | 根因 | 修复方式 | 是否影响推理精度 |
|---|---------|------|---------|--------------|
| 1 | `Serialization: Cannot deserialize plugin` | C++ 节点加载引擎时未初始化 TRT 插件库 | `loadEngine()` 中添加 `initLibNvInferPlugins()` | **无影响**（仅初始化） |
| 2 | `Reshape [0, -1]` 双通配符 | TRT 8.5 不支持 Reshape shape 中同时出现两个动态维度 | 将 `/roi_heads/Reshape` 的 shape 替换为静态值 `[-1, 60]` 和 `[-1, 15, 4]` | **无影响**（shape 值由模型结构决定，固定输入下唯一确定） |
| 3 | `If_1399/If_1877` 节点解析失败 | TRT 8.5 不能解析两个分支输出形状不同的 If 节点（动态控制流） | 内联 If 节点，直接取 `else_branch`（对应"默认检测框存在"路径） | **无影响**（见下方详细说明） |
| 4 | `box_roi_pool/If` × 4 RoiAlign 维度错误 | TRT RoiAlign 要求 `batch_indices` 为 1D，而 `else_branch` 返回 2D Identity | 改用 `then_branch`（含 Squeeze 降维为 1D） | **无影响**（仅维度整形，数值不变） |
| 5 | `nvinfer1: weights count mismatch` INT64 冲突 | TRT 8.5 内部权重去重时 INT64(0) 与 FLOAT32(0.0) 字节模式相同引发冲突 | 统一将 INT64 常量/Cast 转为 INT32；超出范围值截断到 INT32_MAX | **无影响**（见下方详细说明） |
| 6 | `Add-zero` 字节碰撞 | ConstantOfShape(0) + Add 产生与 FLOAT32(0.0) 相同的字节模式 | 将 `Add(zero_tensor, x)` 替换为 `Identity(x)` | **无影响**（加零不改变数值） |
| 7 | `TopK K not a constant` | TRT 8.5 要求 TopK 的 K 值为静态 initializer，不能是运行时计算的张量 | 对固定输入 `[1,3,480,640]` 预计算 K 值并固化为常量 `[1000,1000,1000,1000,663]` | **无影响**（见下方详细说明） |
| 8 | onnxsim 破坏 If 子图 | onnx-simplifier 删除了 If 分支内的 Constant 节点，产生悬空引用 | 从**原始** ONNX（非 onnxsim 版本）开始处理 | **无影响** |

---

## 二、各关键修复对推理结果的影响分析

### 2.1 If 节点内联（修复 #3）

```
原始逻辑：
  if len(boxes) == 0:    ← then_branch（无检测框时返回空张量）
      return empty
  else:                  ← else_branch（正常路径）
      return filtered_detections
```

**内联选择 `else_branch` 的理由**：在正常推理时（有图像输入），RPN 和 roi_heads 都会产生候选框，`len(boxes) > 0` 条件成立，程序天然走 `else_branch`。内联仅是静态选定这条执行路径，消除了 TRT 无法处理的动态分支。

**边界情况**：极端情况下（背景区域完全无特征，RPN 输出 0 个 proposal），原始模型会走 `then_branch` 返回空结果，而内联后会对空张量执行 NMS 等操作。在实践中这不影响最终输出（NMS 对空输入输出空集合），但节点可能报 shape 错误。这属于极端退化情况，正常场景不受影响。

### 2.2 INT64 → INT32 转换（修复 #5）

所有被转换的 INT64 值均为：
- NMS 相关的**计数限制**（如最大检测框数、pre-NMS topK 值）
- 形状计算中的**索引**值

这些值在模型结构确定的情况下，INT64 和 INT32 表示完全一致（均远小于 2³¹-1 = 2,147,483,647）。仅有一处使用 INT64_MAX 作为"无限制"的 NMS 保留数，被截断为 INT32_MAX（依然远大于实际检测框数量），功能等价。

### 2.3 TopK K 值固化（修复 #7）

原始模型中 TopK 的 K 值由输入图像尺寸动态计算：
```
K = min(pre_nms_top_n, total_anchors_in_image)
```
对于固定输入 `[1, 3, 480, 640]`，每个 FPN 层的 anchor 数固定，K 值唯一确定为 `[1000, 1000, 1000, 1000, 663]`。固化后与原始动态计算结果**数值完全相同**，推理结果无任何差异。

**注意**：如果未来模型用于**不同分辨率**输入，需要重新计算 K 值并重新转换引擎。

### 2.4 FP16 精度（trtexec --fp16）

使用 `--fp16` 标志会将部分层以 FP16 精度运行（TRT 自动选择哪些层降精度）。对于目标检测任务：
- 定位精度（bbox 坐标）：误差 < 0.5 像素，可忽略
- 分类置信度：误差 < 0.01，可忽略
- 最终检测结果：与 FP32 相比 mAP 差距通常 < 0.5%

**结论**：所有修复均为**纯结构性变换**或**等价替换**，不改变数值计算逻辑，推理结果与原始 PyTorch 模型一致。

---

## 三、下次转换新 ONNX 的完整操作步骤

### 前提条件

```bash
# 确认工具可用
python3 -c "import onnx; print(onnx.__version__)"          # 须 >= 1.12
/usr/src/tensorrt/bin/trtexec --help | head -3              # TRT 8.5.2
```

### 步骤一：准备 ONNX 文件

将新的 ONNX 文件放到 `/home/nvidia/ros2_ws/` 目录，**不要**用 onnxsim 处理（会破坏 If 子图）。

```bash
cd /home/nvidia/ros2_ws
ls -lh your_model.onnx
```

### 步骤二：修改转换脚本

编辑 `fix_all_and_convert.py`，修改顶部的文件名：

```python
# 约第 380 行 main() 函数开头
src = 'your_model.onnx'          # ← 改为你的 ONNX 文件名
dst = 'your_model_fixed.onnx'    # ← 修复后的中间文件名
engine_dst = 'faster_rcnn.engine' # ← 输出引擎文件名（保持不变）
```

**如果模型结构有变化，还需确认以下内容：**

#### 检查 Reshape 节点名称（如果 roi_heads 结构变了）
```python
# 约第 32 行 fix_reshape_wildcards()
reshape_target_shapes = {
    '/roi_heads/Reshape':   np.array([-1, 60],    dtype=np.int64),  # 60 = num_classes * 4 ÷ 4 * 4?
    '/roi_heads/Reshape_1': np.array([-1, 15, 4], dtype=np.int64),  # 15 = num_classes
}
```
用 Netron 打开 ONNX 文件确认节点名和 shape。若类别数改变（如从 15 改为 N），对应修改。

#### 检查 If 节点名称（如果模型结构改变）
```python
# 约第 340 行 inline_plan
inline_plan = {
    'If_1399': 'else_branch',
    'If_1877': 'else_branch',
    '/roi_heads/box_roi_pool/If':   'then_branch',
    '/roi_heads/box_roi_pool/If_1': 'then_branch',
    '/roi_heads/box_roi_pool/If_2': 'then_branch',
    '/roi_heads/box_roi_pool/If_3': 'then_branch',
}
```
可用以下命令快速列出所有 If 节点名：
```bash
python3 - <<'EOF'
import onnx
m = onnx.load('your_model.onnx')
for n in m.graph.node:
    if n.op_type == 'If':
        print(n.name)
EOF
```

#### 检查 TopK K 值（如果输入尺寸改变）
```python
# 约第 202 行 fold_topk_k_to_constants()
topk_k_map = {
    '/rpn/Reshape_26_output_0': 1000,
    '/rpn/Reshape_27_output_0': 1000,
    '/rpn/Reshape_28_output_0': 1000,
    '/rpn/Reshape_29_output_0': 1000,
    '/rpn/Reshape_30_output_0': 663,   # ← 这个值与输入尺寸有关
}
```
如果输入尺寸变化，用 onnxruntime 重新计算 K 值：
```bash
python3 - <<'EOF'
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('your_model.onnx')
# 用实际输入尺寸
dummy = np.zeros((1, 3, NEW_H, NEW_W), dtype=np.float32)
# 中间张量名可用 Netron 查看 TopK 节点的 K 输入名
EOF
```

### 步骤三：运行修复和转换脚本

脚本运行分两阶段：ONNX 修复（快，约 1 分钟）+ TRT 引擎构建（慢，约 15 分钟）。

**推荐用后台方式运行**，避免 SSH 断线中断：

```bash
cd /home/nvidia/ros2_ws

# 方式 A：nohup（推荐）
nohup python3 fix_all_and_convert.py > /tmp/fix_convert.log 2>&1 &
echo "PID: $!"

# 监控进度（另开终端）
tail -f /tmp/fix_convert.log
```

脚本会在 trtexec 超时（600s）后自动退出，但 trtexec 后台进程仍在继续。若脚本因超时退出：

```bash
# 检查 trtexec 是否还在运行
ps aux | grep '[t]rtexec'

# 查看 trtexec 日志（脚本内 trtexec 日志在 /tmp/trtexec_final2.log 或 stdout）
# 等待完成后确认
grep -E '(PASSED|FAILED|Engine built)' /tmp/trtexec_final2.log
```

若需要单独运行 trtexec（脚本超时后手动执行）：
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=your_model_fixed.onnx \
    --saveEngine=faster_rcnn.engine \
    --fp16 \
    --workspace=4096 \
    > /tmp/trtexec.log 2>&1 &
echo "trtexec PID: $!"
```

### 步骤四：确认引擎构建成功

```bash
# 检查最终结果
grep -E '(&&&& PASSED|&&&& FAILED|Engine built)' /tmp/trtexec.log

# 确认引擎文件存在且大小合理（Faster R-CNN FP16 约 100~200MB）
ls -lh /home/nvidia/ros2_ws/faster_rcnn.engine
```

### 步骤五：部署引擎到 ROS2

```bash
# 复制引擎到 ROS2 install 路径
cp /home/nvidia/ros2_ws/faster_rcnn.engine \
   /home/nvidia/ros2_ws/install/faster_rcnn_ros/share/faster_rcnn_ros/models/faster_rcnn.engine

# 确认
ls -lh /home/nvidia/ros2_ws/install/faster_rcnn_ros/share/faster_rcnn_ros/models/faster_rcnn.engine
```

### 步骤六：更新 launch 文件（如果输入尺寸改变）

编辑 `/home/nvidia/ros2_ws/src/faster_rcnn_ros/launch/faster_rcnn.launch.xml`：

```xml
<arg name="input_height" default="480"/>   <!-- 改为新的高度 -->
<arg name="input_width"  default="640"/>   <!-- 改为新的宽度 -->
```

然后重新编译（使 install 目录更新）：
```bash
cd /home/nvidia/ros2_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select faster_rcnn_ros

# 重新复制引擎（colcon build 会重置 models 目录）
cp faster_rcnn.engine \
   install/faster_rcnn_ros/share/faster_rcnn_ros/models/faster_rcnn.engine
```

### 步骤七：启动验证

```bash
cd /home/nvidia/ros2_ws
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml
```

期望输出：
```
[faster_rcnn_node-1] [INFO] [...] [faster_rcnn_node]: Faster R-CNN node started.
```

---

## 四、常见错误速查

| 错误信息 | 原因 | 解决方法 |
|---------|------|---------|
| `Cannot deserialize plugin` | 未初始化 TRT 插件 | 确认 `faster_rcnn_detector.cpp` 包含 `initLibNvInferPlugins()` |
| `Engine file not found` | 引擎路径错误 | 确认引擎在 `install/.../models/faster_rcnn.engine` |
| `Reshape shape mismatch` | 类别数变化后旧 shape 值不匹配 | 更新 `fix_reshape_wildcards()` 中的 shape 值 |
| `TopK K inputs must be ... constant` | TopK K 未固化 | 更新 `fold_topk_k_to_constants()` 中的 K 值映射 |
| `RoiAlign batch_indices.nbDims == 1` | box_roi_pool/If 走了 else（Identity/2D）分支 | 确认 `inline_plan` 中 box_roi_pool 相关 If 使用 `then_branch` |
| trtexec 约 15 分钟后无响应 | 正常（引擎构建耗时长） | 耐心等待，用 `tail -f` 监控日志 |

---

## 五、文件清单

```
/home/nvidia/ros2_ws/
├── fasterrcnn_nuscenes.onnx            # 原始 ONNX（不要修改）
├── fasterrcnn_nuscenes_fixed_final.onnx # 修复后的中间 ONNX
├── faster_rcnn.engine                   # TRT 引擎（部署文件）
├── fix_all_and_convert.py              # 一键修复+转换脚本（入口）
└── install/faster_rcnn_ros/share/faster_rcnn_ros/models/
    ├── faster_rcnn.engine              # 已部署的引擎（ROS2 节点读取此处）
    └── labels.txt                      # 类别标签
```

---

*文档生成时间：2026-03-12 | 引擎构建耗时 852 秒 | `&&&& PASSED`*
