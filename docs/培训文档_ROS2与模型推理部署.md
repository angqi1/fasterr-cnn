# Faster R-CNN 目标检测系统 — 培训手册

**平台**：NVIDIA Jetson AGX Orin · JetPack 5.1.2 · ROS2 Foxy · TensorRT 8.5  
**适合人群**：对 ROS2 和深度学习模型推理有初步了解意愿的工程师  
**阅读时长**：约 75 分钟

---

## 目录

1. [ROS2 核心概念](#1-ros2-核心概念)
2. [项目总体架构](#2-项目总体架构)
3. [环境安装与初始化](#3-环境安装与初始化)
4. [代码文件详解](#4-代码文件详解)
5. [模型推理框架（TensorRT）](#5-模型推理框架tensorrt)
6. [数据流转全流程](#6-数据流转全流程)
7. [关键函数逐行解读](#7-关键函数逐行解读)
8. [常用操作命令](#8-常用操作命令)
9. [性能评估结果](#9-性能评估结果)
10. [常见问题排查](#10-常见问题排查)
11. [ROS2 对多传感器系统的优势与必要性](#11-ros2-对多传感器系统的优势与必要性)
12. [单引擎与双引擎模式详解](#12-单引擎与双引擎模式详解)
13. [培训重点讲解指南](#13-培训重点讲解指南)

---

## 1. ROS2 核心概念

### 1.1 什么是 ROS2？

ROS2（Robot Operating System 2）是一个**机器人中间件框架**，本质上是一套进程间通信（IPC）规范和工具链。它不是操作系统，而是运行在 Linux 之上的软件层。

```
┌──────────────────────────────────────────────────────┐
│                   用户代码（节点）                    │
├──────────────────────────────────────────────────────┤
│           ROS2 API（rclcpp / rclpy）                 │
├──────────────────────────────────────────────────────┤
│      DDS 中间件（Fast-DDS / Cyclone DDS）             │  ← 负责实际网络通信
├──────────────────────────────────────────────────────┤
│             Linux 操作系统                            │
└──────────────────────────────────────────────────────┘
```

**ROS2 vs ROS1 的主要区别：**
| 特性 | ROS1 | ROS2 |
|------|------|------|
| 通信协议 | 自研 TCPROS | 基于 DDS 标准 |
| 多机通信 | 需要手动配置 Master | 自动发现，无需 rosmaster |
| 实时性 | 无 | 支持实时调度 |
| 安全性 | 无加密 | 支持 SROS2 加密 |
| Python 版本 | Python 2/3 | Python 3 |

### 1.2 核心概念：节点（Node）

**节点**是 ROS2 中最基本的执行单元，每个节点是一个独立的进程（或线程），负责完成一项特定任务。

```
节点 A（相机驱动）  ──发布消息──▶  节点 B（检测器）  ──发布消息──▶  节点 C（可视化）
```

本项目中只有 **1 个节点**：`faster_rcnn_node`，它自己完成从接收图像到输出检测结果的全部工作。

### 1.3 核心概念：话题（Topic）

话题是 ROS2 进程间**异步通信**的管道，遵循"发布-订阅"模型：

```
发布者（Publisher）  ──[话题名称/消息类型]──▶  订阅者（Subscriber）
                                ↑
                     多个订阅者可以同时接收
```

本项目涉及的话题：

| 话题名称 | 消息类型 | 方向 | 说明 |
|---------|---------|------|------|
| `/detectnet/image_in` | `sensor_msgs/msg/Image` | 输入 | 原始相机图像 |
| `/detectnet/overlay` | `sensor_msgs/msg/Image` | 输出 | 叠加检测框的结果图像 |

### 1.4 核心概念：参数（Parameter）

节点可以声明参数，在启动时从外部注入配置值，无需修改代码：

```bash
# 启动节点，并通过命令行设置参数
ros2 run faster_rcnn_ros faster_rcnn_node --ros-args \
  -p threshold:=0.5 \
  -p engine_path:=/path/to/model.engine
```

### 1.5 核心概念：Launch 文件

Launch 文件是 XML 格式的批量启动脚本，可以：
- 一次启动多个节点
- 为每个节点设置参数默认值
- 定义参数别名（`<arg>`）方便外部覆盖

```xml
<!-- 示例：声明一个可被外部覆盖的参数 -->
<arg name="threshold" default="0.5"/>

<!-- 启动节点并将参数传入 -->
<node pkg="包名" exec="节点可执行文件名" output="screen">
    <param name="threshold" value="$(var threshold)"/>
</node>
```

### 1.6 核心概念：消息类型（Message）

ROS2 使用结构化消息在节点间传递数据。本项目使用的消息：

**`sensor_msgs/msg/Image`** — 图像消息结构：
```
Header header          ← 时间戳 + 坐标系 ID
uint32 height          ← 图像高度（像素）
uint32 width           ← 图像宽度（像素）
string encoding        ← 编码格式，如 "bgr8", "rgb8"
uint8[] data           ← 原始像素字节数组（height × width × 3）
```

### 1.7 工作空间（Workspace）结构

ROS2 工作空间是一个约定的目录结构：

```
ros2_ws/                  ← 工作空间根目录
├── src/                  ← 源代码（手动放置）
│   └── faster_rcnn_ros/  ← 一个 ROS2 功能包（Package）
├── build/                ← 编译中间文件（colcon 自动生成）
├── install/              ← 安装后的可执行文件和资源（colcon 自动生成）
└── log/                  ← 构建日志（colcon 自动生成）
```

**功能包（Package）** 是 ROS2 的代码组织单位，每个包必须包含：
- `package.xml` — 包的元信息（名称、版本、依赖）
- `CMakeLists.txt` — C++ 包的构建规则（Python 包用 `setup.py`）

---

## 2. 项目总体架构

### 2.1 系统层次图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户输入层                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │ 相机/话题    │  │  图片文件      │  │  视频文件 / 图片目录         │  │
│  │ (实时模式)   │  │  (离线模式)   │  │  (批量处理模式)              │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────┬───────────────┘  │
└─────────┼────────────────┼───────────────────────┼─────────────────┘
          │                │                         │
          ▼                ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ROS2 节点层（C++ 代码）                            │
│                                                                      │
│  faster_rcnn_node.cpp                                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  topicCallback()         fileTimerCallback()                  │   │
│  │       ↓                          ↓                           │   │
│  │            runInference(frame)                                │   │
│  │                   ↓                                          │   │
│  │     filterAndNMS(raw, car_thr, ped_thr)                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   检测器封装层（C++ 代码）                            │
│                                                                      │
│  faster_rcnn_detector.cpp / .hpp                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  infer(image) → inferAsync() + syncAndCollect()              │   │
│  │       ↓               ↓                  ↓                   │   │
│  │   预处理          H2D 拷贝           D2H 拷贝                 │   │
│  │  (blobFromImage)  (CPU→GPU)     (GPU→CPU, 缩放坐标)          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   GPU 推理层（TensorRT）                              │
│                                                                      │
│  faster_rcnn_500.engine（FP16 量化，500×1242 分辨率）                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  image [1,3,500,1242]                                        │   │
│  │       ↓                                                      │   │
│  │  [FPN 特征提取] → [RPN 候选框] → [ROI Pooling] → [分类/回归] │   │
│  │       ↓                                                      │   │
│  │  scores [-1]  labels [-1]  boxes [-1, 4]                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 两种运行模式

```
               ┌── 话题模式（Topic Mode）────────────────────┐
               │                                              │
               │  相机节点 ──▶ /detectnet/image_in ──▶ 节点  │
  启动节点 ──▶ │                                    ──▶ /detectnet/overlay ──▶ RViz
               │                                              │
               └── 文件模式（File Mode）────────────────────┐│
                    input_path 非空时自动切换                 │
                    ├─ 单张图片 → 推理 → 保存 result.jpg      │
                    ├─ 视频文件 → 逐帧推理 → 保存 result.mp4  │
                    └─ 图片目录 → 批量推理 → 保存到输出目录   │
                                                              │
                                                             ┘
```

---

## 3. 环境安装与初始化

### 3.1 前置依赖（Jetson 平台已预装）

| 软件 | 版本 | 说明 |
|------|------|------|
| JetPack | 5.1.2 | 包含 CUDA 11.4、cuDNN 8.6 |
| TensorRT | 8.5.2 | 神经网络推理加速框架 |
| OpenCV | 4.5+ | 图像处理库（含 cv_bridge） |
| ROS2 Foxy | 2021.06 | 机器人操作系统（Ubuntu 20.04 LTS 对应版本）|
| Python | 3.8+ | Python 推理工具脚本 |

### 3.2 安装 ROS2 Foxy（Ubuntu 20.04）

```bash
# 1. 添加 ROS2 APT 源
sudo apt install software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list

# 2. 安装 ROS2 基础版
sudo apt update
sudo apt install ros-foxy-ros-base python3-colcon-common-extensions

# 3. 安装本项目额外依赖
sudo apt install ros-foxy-cv-bridge ros-foxy-sensor-msgs

# 4. 每次打开新终端时需要 source（或写入 ~/.bashrc）
source /opt/ros/foxy/setup.bash
```

### 3.3 首次编译项目

```bash
cd /home/nvidia/ros2_ws

# 安装包依赖
rosdep install --from-paths src --ignore-src -r -y

# 编译
colcon build --packages-select faster_rcnn_ros

# 激活安装环境（每次新终端都需执行）
source install/setup.bash
```

> **注意**：编译完成后，修改 C++ 源代码必须重新 `colcon build` 才生效；Launch 文件和 Python 脚本修改后 **无需重新编译**（install 目录通过符号链接指向源码目录）。

### 3.4 GPU 时钟锁频（重要！）

Jetson 默认以最低功耗模式运行 GPU，需手动锁频以获得最佳性能：

```bash
sudo jetson_clocks      # 锁定 CPU/GPU 至最高频率（930 MHz）

# 验证：GPU 应显示 930 MHz（不是 306 MHz）
sudo tegrastats | grep GR3D_FREQ
```

---

## 4. 代码文件详解

### 4.1 项目文件树（精简版）

```
ros2_ws/
├── src/faster_rcnn_ros/             ← ROS2 功能包
│   ├── package.xml                  ← 包元信息
│   ├── CMakeLists.txt               ← 编译规则
│   │
│   ├── include/faster_rcnn_ros/
│   │   └── faster_rcnn_detector.hpp ← 检测器类声明（接口定义）
│   │
│   ├── src/
│   │   ├── faster_rcnn_detector.cpp ← 检测器实现（TRT 调用核心）
│   │   └── faster_rcnn_node.cpp     ← ROS2 节点（业务逻辑）
│   │
│   ├── launch/
│   │   └── faster_rcnn.launch.xml   ← 启动配置文件
│   │
│   └── models/
│       ├── labels.txt               ← 11 类别名称
│       ├── faster_rcnn_500.engine   ← FP16 推理引擎（主用，115MB）
│       ├── faster_rcnn_500_int8.engine ← INT8 引擎（压缩版，58MB）
│       ├── fasterrcnn_nuscenes_fixed_500x1242.onnx ← 修复后的 ONNX
│       ├── build_engine.py          ← ONNX→TRT 引擎构建脚本
│       └── build_all_precisions.sh  ← 一键构建 FP16/INT8 引擎
│
├── gt_eval_all.py                   ← 精度评估：4 配置 vs KITTI GT
├── run_comparison.py                ← 调用 C++ 节点批量对比
├── compare_gt.py                    ← ONNX Runtime 推理 + GT 对比
├── download_kitti_subset.py         ← KITTI 数据集下载工具
├── test_pub.py                      ← ROS2 图像话题测试发布器
│
├── PIPELINE.md                      ← 推理流程详细文档
├── ONNX_TO_TRT_GUIDE.md             ← ONNX→TRT 转换指南
│
└── test_images/
    ├── kitti_100/images/            ← 100 张 KITTI 测试图片
    ├── kitti_100/labels/            ← 对应的 KITTI GT 标注
    └── compare_results/             ← 4 配置对比结果图片
```

---

### 4.2 `package.xml` — 包描述文件

```xml
<package format="3">
  <name>faster_rcnn_ros</name>       <!-- 包名，必须与目录名一致 -->
  <version>0.0.1</version>
  <description>Faster R-CNN ROS2 目标检测节点</description>

  <!-- 编译时依赖（编译阶段需要） -->
  <build_depend>rclcpp</build_depend>
  <build_depend>sensor_msgs</build_depend>
  <build_depend>cv_bridge</build_depend>

  <!-- 运行时依赖（运行阶段需要） -->
  <exec_depend>rclcpp</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>

  <buildtool_depend>ament_cmake</buildtool_depend>
</package>
```

---

### 4.3 `CMakeLists.txt` — 编译规则

CMakeLists.txt 是 CMake 构建系统的配置文件，告诉编译器：
1. 需要找哪些库
2. 如何编译源文件
3. 把什么安装到哪里

```cmake
cmake_minimum_required(VERSION 3.8)
project(faster_rcnn_ros)

# ── 查找依赖库 ──────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)      # ROS2 构建工具
find_package(rclcpp REQUIRED)           # ROS2 C++ 客户端库
find_package(sensor_msgs REQUIRED)       # 传感器消息定义
find_package(cv_bridge REQUIRED)         # OpenCV ↔ ROS2 图像转换
find_package(OpenCV REQUIRED)            # OpenCV 图像处理
find_package(CUDA REQUIRED)              # CUDA 并行计算

# ── 编译可执行文件 ──────────────────────────────────────────────────────
add_executable(faster_rcnn_node
    src/faster_rcnn_node.cpp
    src/faster_rcnn_detector.cpp     # 两个 .cpp 文件编译为一个可执行文件
)

# ── 链接共享库 ──────────────────────────────────────────────────────────
target_link_libraries(faster_rcnn_node
    nvinfer           # TensorRT 推理引擎
    nvinfer_plugin    # TensorRT 插件（NMS 等算子）
    cudart            # CUDA 运行时
    ${OpenCV_LIBS}    # OpenCV
)

# ── 安装：把编译产物复制到 install/ 目录 ────────────────────────────────
install(TARGETS faster_rcnn_node
    DESTINATION lib/${PROJECT_NAME})    # 可执行文件

install(DIRECTORY launch models         # launch 和模型文件
    DESTINATION share/${PROJECT_NAME})
```

---

### 4.4 `faster_rcnn_detector.hpp` — 检测器接口声明

头文件定义了**数据结构**和**类的公共接口**，是使用者需要了解的 API：

```cpp
// ── 单个检测结果 ─────────────────────────────────────────────────────
struct Detection {
    cv::Rect box;        // 检测框：左上角坐标(x,y) + 宽高(w,h)
    float confidence;    // 置信度分数 [0.0, 1.0]
    int class_id;        // 类别ID: 0=背景, 1=car, ..., 6=pedestrian ...
};

// ── TensorRT 动态输出分配器 ──────────────────────────────────────────
// TRT 8.5+ 引入：检测框数量是动态的(-1)，需要动态分配 GPU 显存
class OutputAllocator : public nvinfer1::IOutputAllocator {
public:
    // TRT 在每次推理前调用此函数，按需分配输出缓冲区
    void* reallocateOutput(char const* name, void* ptr,
                           uint64_t size, uint64_t alignment) override;
    // 推理完成后，TRT 调用此函数告知实际输出形状
    void notifyShape(char const* name, nvinfer1::Dims const& dims) override;

    void*  buffer = nullptr;      // GPU 显存指针
    size_t bufferSize = 0;        // 当前分配大小（按需扩容）
    nvinfer1::Dims outputDims;    // 实际输出形状（如检测框数量）
};

// ── 检测器类 ─────────────────────────────────────────────────────────
class FasterRCNNDetector {
public:
    // 构造函数：加载引擎文件，分配显存，初始化 CUDA 流
    FasterRCNNDetector(const std::string& engine_path,
                       const std::string& labels_path,
                       int input_h, int input_w);
    ~FasterRCNNDetector();

    // 同步推理（最常用）：输入图像，返回检测结果
    std::vector<Detection> infer(const cv::Mat& image, float threshold = 0.5f);

    // 异步推理接口（高级用法，用于双引擎并行）
    void inferAsync(const cv::Mat& image);
    std::vector<Detection> syncAndCollect(const cv::Mat& image, float threshold);

    const std::vector<std::string>& getClassNames() const;
};
```

---

### 4.5 `faster_rcnn_detector.cpp` — 检测器实现

这是整个项目最核心的文件，负责与 TensorRT 引擎交互。

#### 构造函数：初始化

```cpp
FasterRCNNDetector::FasterRCNNDetector(
    const std::string& engine_path,   // 引擎文件路径
    const std::string& labels_path,   // 标签文件路径
    int input_h, int input_w)         // 推理分辨率
{
    // 1. 读取标签文件
    std::ifstream label_file(labels_path);
    // 将每行类别名称读入 class_names_ 向量

    // 2. 加载 TRT 引擎
    loadEngine(engine_path);

    // 3. 创建执行上下文（可理解为引擎的"执行实例"）
    context_.reset(engine_->createExecutionContext());

    // 4. 固定输入形状
    context_->setInputShape("image", Dims4{1, 3, input_h_, input_w_});

    // 5. 注册动态输出分配器（scores/labels/boxes 数量动态变化）
    scores_alloc_ = std::make_unique<OutputAllocator>();
    context_->setOutputAllocator("scores", scores_alloc_.get());
    // labels/boxes 同理...

    // 6. 分配输入显存（固定大小：3×H×W×sizeof(float)）
    cudaMalloc(&d_input_, 3 * input_h_ * input_w_ * sizeof(float));
    context_->setTensorAddress("image", d_input_);

    // 7. 创建 CUDA 流（允许异步执行）
    cudaStreamCreate(&stream_);
}
```

#### `inferAsync()` — 异步推理入队

```cpp
void FasterRCNNDetector::inferAsync(const cv::Mat& image) {
    // ① 预处理：将 BGR 图像转为模型所需格式
    //   - 缩放到输入分辨率（500×1242）
    //   - BGR → RGB（swapRB=true）
    //   - 归一化到 [0,1]（scale=1/255）
    //   - 重排维度：HWC → CHW（channels first，PyTorch 格式）
    //   - 打包为连续内存 blob (1, 3, H, W)
    cv::dnn::blobFromImage(image, blob_,
                           1.0 / 255.0,          // 归一化系数
                           cv::Size(input_w_, input_h_),  // 目标尺寸
                           cv::Scalar(),          // 均值（不减均值）
                           /*swapRB=*/true,       // BGR→RGB
                           /*crop=*/false);       // 不裁剪，全图缩放

    // ② H2D：将预处理后的数据从 CPU 内存拷贝到 GPU 显存
    cudaMemcpyAsync(d_input_, blob_.data, input_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // ③ 将推理任务加入 GPU 执行队列（非阻塞，立即返回）
    context_->enqueueV3(stream_);
    // ↑ CPU 不等待 GPU 完成，可以继续做其他事（双引擎并行的关键）
}
```

#### `syncAndCollect()` — 等待并收集结果

```cpp
std::vector<Detection> FasterRCNNDetector::syncAndCollect(
    const cv::Mat& image, float threshold)
{
    // ① 等待 GPU 上的推理计算真正完成
    cudaStreamSynchronize(stream_);

    // ② 读取动态输出的实际数量
    int n_dets = scores_alloc_->outputDims.d[0];
    if (n_dets <= 0) return {};

    // ③ D2H：从 GPU 显存拷贝结果到 CPU 内存
    std::vector<float> h_scores(n_dets);
    std::vector<int>   h_labels(n_dets);
    std::vector<float> h_boxes(n_dets * 4);
    cudaMemcpy(h_scores.data(), scores_alloc_->buffer, ...);
    // labels/boxes 同理...

    // ④ 坐标缩放：引擎输出的坐标是基于输入分辨率（500×1242）
    //            需要换算回原始图像分辨率（如 370×1224）
    float sx = (float)image.cols / input_w_;  // 宽度缩放比
    float sy = (float)image.rows / input_h_;  // 高度缩放比

    // ⑤ 后处理：阈值过滤 + 边界约束 + 封装为 Detection
    for (int i = 0; i < n_dets; ++i) {
        if (h_scores[i] < threshold) continue;  // 低置信度丢弃
        int x1 = h_boxes[i*4+0] * sx;           // 缩放 x1
        // ... 同理处理 y1, x2, y2
        detections.push_back({cv::Rect(...), h_scores[i], h_labels[i]});
    }
    return detections;
}
```

---

### 4.6 `faster_rcnn_node.cpp` — ROS2 节点

ROS2 节点负责**业务调度**，不直接操作 GPU。

#### 节点构造函数：初始化

```cpp
FasterRCNNNode() : Node("faster_rcnn_node") {
    // 1. 声明全部参数（类型、默认值）
    declare_parameter<std::string>("engine_path", "");
    declare_parameter<double>("threshold", 0.5);
    declare_parameter<double>("ped_threshold", 0.5);
    // ... 其他参数

    // 2. 读取参数值
    auto engine_path = get_parameter("engine_path").as_string();
    threshold_       = get_parameter("threshold").as_double();

    // 3. 创建检测器（加载 TRT 引擎）
    detector_ = std::make_unique<FasterRCNNDetector>(
        engine_path, labels_path, input_h, input_w);

    // 4. 根据 input_path 是否为空，决定运行模式
    if (input_path_.empty()) {
        // 话题模式：创建订阅者
        subscription_ = create_subscription<sensor_msgs::msg::Image>(
            input_topic, 10,
            std::bind(&FasterRCNNNode::topicCallback, this, _1));
    } else {
        // 文件模式：初始化文件处理器
        initFileMode();
    }
}
```

#### `runInference()` — 统一推理入口

```cpp
std::vector<Detection> runInference(const cv::Mat& frame) {
    // 快速路径：单引擎 + 统一阈值（当前默认配置）
    if (!detector2_ && ped_threshold_ >= threshold_) {
        return detector_->infer(frame, static_cast<float>(threshold_));
    }

    // 双引擎路径：两个引擎同时异步入队，再依次等待（隐式并行）
    float min_thr = std::min(threshold_, ped_threshold_);
    detector_->inferAsync(frame);          // 引擎1 入队，不等待
    if (detector2_) detector2_->inferAsync(frame);  // 引擎2 立即入队

    auto raw = detector_->syncAndCollect(frame, min_thr);  // 等待引擎1
    if (detector2_) {
        auto dets2 = detector2_->syncAndCollect(frame, min_thr);  // 等待引擎2
        raw.insert(raw.end(), dets2.begin(), dets2.end());  // 合并结果
    }

    // 应用类别专属阈值 + NMS
    return filterAndNMS(raw, threshold_, ped_threshold_);
}
```

#### `filterAndNMS()` — 类别专属阈值过滤 + NMS

```cpp
static std::vector<Detection> filterAndNMS(
    const std::vector<Detection>& raw,
    float car_thr,    // 车辆类阈值（默认 0.5）
    float ped_thr)    // 行人/摩托/自行车阈值（默认 0.5）
{
    // 按类别ID判断应用哪个阈值
    auto classThreshold = [&](int cls) -> float {
        return (cls == 6 || cls == 7 || cls == 8) ? ped_thr : car_thr;
    };

    // 按类别分组，过滤低置信度检测
    std::map<int, pair<vector<Rect>, vector<float>>> by_class;
    for (const auto& d : raw) {
        if (d.confidence >= classThreshold(d.class_id)) {
            by_class[d.class_id].first.push_back(d.box);
            by_class[d.class_id].second.push_back(d.confidence);
        }
    }

    // 每个类别独立 NMS（非极大值抑制，去除重叠框）
    for (auto& [cls, bscore] : by_class) {
        std::vector<int> nms_idx;
        cv::dnn::NMSBoxes(bscore.first,   // 候选框列表
                          bscore.second,  // 对应置信度
                          0.0f,           // score_threshold（已在上面过滤）
                          0.5f,           // IoU 阈值（重叠>50%则抑制）
                          nms_idx);
        // 保留 NMS 后的框
    }
}
```

> **NMS 原理简介**：当多个框检测到同一个目标时，NMS 只保留置信度最高的那个，抑制与它重叠超过 IoU 阈值（0.5）的其他框。

---

### 4.7 `faster_rcnn.launch.xml` — 启动配置文件

```xml
<launch>
    <!-- ① 声明可外部覆盖的参数（arg = 外部参数） -->
    <arg name="engine_path"
         default="$(find-pkg-share faster_rcnn_ros)/models/faster_rcnn_500.engine"/>
    <!-- find-pkg-share 是 ROS2 的路径宏，解析为 install/ 下的 share 目录 -->

    <arg name="threshold"    default="0.5"/>      <!-- 车辆检测阈值 -->
    <arg name="ped_threshold" default="0.5"/>     <!-- 行人检测阈值 -->
    <arg name="input_path"   default=""/>         <!-- 空 = 话题模式 -->
    <arg name="output_path"  default=""/>         <!-- 空 = 不保存 -->

    <!-- ② 启动节点，将 arg 绑定到 param（param = 节点内部参数） -->
    <node pkg="faster_rcnn_ros" exec="faster_rcnn_node" output="screen">
        <param name="engine_path" value="$(var engine_path)"/>
        <param name="threshold"   value="$(var threshold)"/>
        <!-- ... 其他参数绑定 -->
    </node>
</launch>
```

---

### 4.8 `build_engine.py` — ONNX 到 TRT 引擎构建

此脚本解决 PyTorch 导出的 ONNX 与 TensorRT 8.5 的兼容性问题：

**问题原因**：PyTorch 的 Faster R-CNN 使用了动态控制流（`if` 节点）和 INT64 算子，而 TRT 8.5 不支持这些特性。

**修复步骤**：

```
ONNX（PyTorch 原始）
     ↓
① 修复 Reshape 通配符（-1 → 具体数值）
     ↓
② 内联 If 条件节点（展开为静态图）
     ↓
③ 固化 TopK 的 K 值（动态 → 静态常量）
     ↓
④ 折叠 ConstantOfShape 节点
     ↓
⑤ INT64 → INT32 类型转换
     ↓
⑥ 拓扑排序（确保节点执行顺序正确）
     ↓
ONNX（TRT 兼容版）
     ↓
trtexec --fp16 → .engine 文件
```

---

## 5. 模型推理框架（TensorRT）

### 5.1 什么是 TensorRT？

TensorRT 是 NVIDIA 开发的**深度学习推理加速框架**，它通过以下手段大幅加速神经网络推理：

| 优化技术 | 说明 |
|---------|------|
| **层融合** (Layer Fusion) | 将相邻的 Conv+BN+ReLU 合并为单个 GPU 算子，减少内存读写 |
| **精度量化** (Quantization) | FP32→FP16/INT8，减少计算量，加快速度 |
| **显存优化** | 分析整个图的数据依赖，最小化 GPU 显存占用 |
| **算子选择** | 针对当前 GPU 型号选择最优 CUDA 内核实现 |

### 5.2 推理精度对比

| 精度 | 存储大小 | 引擎文件大小 | 推理延迟 | 精度影响 |
|------|---------|------------|---------|---------|
| FP32 | 4 字节/参数 | ~230 MB | ~120ms | 基准 |
| **FP16** | 2 字节/参数 | **115 MB** | **~70ms** | 损失<0.1% |
| INT8 | 1 字节/参数 | **58 MB** | **~63ms** | 损失~1.2% Recall |

### 5.3 TRT 工作流程

```
                    【一次性构建（build 阶段）】
           ONNX 模型
               ↓
         TRT Builder（分析+优化图）
               ↓ 可能需要几分钟到十几分钟
         .engine 文件（序列化的优化计划，特定于当前 GPU）


                    【每次推理（runtime 阶段）】
           .engine 文件（二进制）
               ↓ deserialize（反序列化，<1s）
           ICudaEngine 对象（内存中）
               ↓ createExecutionContext
           IExecutionContext 对象（推理实例）
               ↓
           enqueueV3(stream)  ←── 输入数据（GPU 显存）
               ↓
           输出数据（scores, labels, boxes）
```

### 5.4 模型结构（Faster R-CNN）

```
输入图像 (1, 3, 500, 1242)
       ↓
┌──────────────────────┐
│  ResNet50 骨干网络   │  特征提取（共享特征图）
│  (FPN 多尺度特征)    │  输出：P2/P3/P4/P5/P6 共 5 个尺度
└──────────┬───────────┘
           │ 5 个尺度特征图
           ▼
┌──────────────────────┐
│  区域提议网络(RPN)   │  生成候选框（Proposals）
│  每个尺度 ~1000 个   │  共 ~5000 个候选框
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ROI Align           │  从特征图上截取每个候选区域
│  (RoI Pooling 改进)  │  双线性插值，对齐特征
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  分类头 + 回归头     │  最终输出
│  fc6 → fc7 → cls/reg│  cls: 11类得分，reg: 精确坐标
└──────────┬───────────┘
           │
           ▼
输出：scores[-1]  labels[-1]  boxes[-1,4]
     （动态数量，取决于 RPN 阶段通过的候选框数量）
```

### 5.5 类别列表（labels.txt）

```
ID  类别名称              中文名     检测阈值
0   background            背景       -
1   car                   汽车       threshold（默认0.5）
2   truck                 货车       threshold
3   bus                   公交车     threshold
4   trailer               拖车       threshold
5   construction_vehicle  工程车     threshold
6   pedestrian            行人       ped_threshold（默认0.5）
7   motorcycle            摩托车     ped_threshold
8   bicycle               自行车     ped_threshold
9   traffic_cone          交通锥     threshold
10  barrier               护栏       threshold
```

---

## 6. 数据流转全流程

### 6.1 文件模式（图片目录）完整流程

```
用户命令行
ros2 launch faster_rcnn.launch.xml input_path:=./test_images output_path:=./results
                    │
                    ▼
          ROS2 节点初始化
          FasterRCNNNode 构造函数
                    │
                    ├── loadEngine("faster_rcnn_500.engine")
                    │   └── TRT 反序列化：~0.5s，分配 GPU 显存
                    │
                    ├── setInputShape("image", [1,3,500,1242])
                    │
                    └── initFileMode()
                        └── 扫描目录，生成 image_files_ 列表
                            创建 0ms 定时器（立即开始处理）

（每张图片触发一次 Timer 回调）
                    ↓
          fileTimerCallback()
          ┌─────────────────────────────────────────────────────────────┐
          │ ① 读取图片                                                   │
          │    frame = cv::imread("000000.png")                         │
          │    shape: (370, 1224, 3)  uint8  BGR                        │
          │                                                              │
          │ ② 记录开始时间                                               │
          │    t0 = steady_clock::now()                                  │
          │                                                              │
          │ ③ 调用推理                                                   │
          │    detections = runInference(frame)                          │
          │         │                                                    │
          │         └──→ detector_->infer(frame, 0.5)                   │
          │               │                                              │
          │               ├── inferAsync(frame)                         │
          │               │    ├── blobFromImage→ shape:[1,3,500,1242]  │
          │               │    ├── cudaMemcpyAsync(CPU→GPU, ~3.7MB)     │
          │               │    └── enqueueV3(stream)  ← GPU 开始执行    │
          │               │                                              │
          │               └── syncAndCollect(frame, 0.5)                │
          │                    ├── cudaStreamSynchronize ← 等待 GPU     │
          │                    ├── 读取 n_dets（比如 3）                │
          │                    ├── cudaMemcpy(GPU→CPU, scores/labels/boxes) │
          │                    ├── 坐标缩放：×(1224/1242), ×(370/500)   │
          │                    └── 过滤 score<0.5，返回 vector<Detection>│
          │                                                              │
          │ ④ 计算耗时                                                   │
          │    elapsed_ms = (now - t0).count()  → 约 68ms               │
          │                                                              │
          │ ⑤ 绘制检测框                                                 │
          │    drawDetections(overlay, detections, class_names)          │
          │    每个 Detection 绘制：绿色矩形框 + "car: 0.87" 文字标签   │
          │                                                              │
          │ ⑥ 打印日志                                                   │
          │    [000000.png] 3 个目标 | 68.2 ms                          │
          │                                                              │
          │ ⑦ 保存结果图片                                               │
          │    cv::imwrite("results/result_000000.png", overlay)        │
          │                                                              │
          │ ⑧ 发布到 ROS2 话题（可选）                                  │
          │    overlay_pub_->publish(cv_bridge::CvImage(...))            │
          └─────────────────────────────────────────────────────────────┘
                    ↓
          file_index_++ 继续下一张
          …（共 100 张）
                    ↓
          所有图片处理完成
          file_timer_->cancel()
          rclcpp::shutdown()
```

### 6.2 话题模式（实时相机流）完整流程

```
相机驱动节点                    faster_rcnn_node
      │                               │
      │ 每帧图像（约30fps）            │
      │──publish("/detectnet/image_in")──▶│
      │                               │
      │                    topicCallback(msg)
      │                               │
      │                    ① 解码 ROS2 图像消息
      │                       cv_bridge::toCvCopy(msg, "bgr8")
      │                       → cv::Mat frame
      │                               │
      │                    ② runInference(frame)
      │                       [与文件模式相同的推理流程]
      │                               │
      │                    ③ 绘制检测框
      │                               │
      │                    ④ 发布结果
      │        overlay_pub_->publish(...)
      │                    │
      ◀───────────────────────────────│
      │                  /detectnet/overlay (带框图像)
      │                               │
   RViz 可视化
   （订阅 /detectnet/overlay）
```

### 6.3 GPU 内存数据流

```
CPU 内存                           GPU 显存
─────────────────────────────────────────────────────────────
cv::Mat frame
(370, 1224, 3) uint8 BGR
      │
      │ blobFromImage（缩放+归一化+转置）
      ▼
blob [1,3,500,1242] float32 RGB
      │
      │ cudaMemcpyAsync (HostToDevice)
      │ ~3.7 MB 数据传输（约 0.5ms）
      ▼
                              d_input_ [1,3,500,1242] float32
                                    │
                                    │ enqueueV3（GPU 上运行 ~60ms）
                                    ▼
                              scores_alloc_->buffer [-1] float32
                              labels_alloc_->buffer [-1] int32
                              boxes_alloc_->buffer  [-1,4] float32
                                    │
      │ cudaMemcpy (DeviceToHost)   │
      │ ~几KB 数据传输               │
      ◀────────────────────────────┘
h_scores [N] float32
h_labels [N] int32
h_boxes  [N,4] float32
      │
      │ 坐标缩放 + 阈值过滤
      ▼
vector<Detection> （返回给调用方）
```

---

## 7. 关键函数逐行解读

### 7.1 预处理：`cv::dnn::blobFromImage`

这是预处理最关键的一行，与模型训练时的预处理必须完全一致：

```cpp
cv::dnn::blobFromImage(
    image,              // 输入：BGR uint8 矩阵
    blob_,              // 输出：NCHW float32 矩阵
    1.0 / 255.0,        // scale：每个像素值乘以此系数（归一化到[0,1]）
    cv::Size(input_w_, input_h_),  // 目标尺寸：宽×高
    cv::Scalar(),       // mean：减去均值（空=不减）
    /*swapRB=*/true,    // BGR→RGB（模型用RGB训练）
    /*crop=*/false      // 不裁剪，直接拉伸缩放
);
// 输出 blob_ 形状：[1, 3, input_h_, input_w_]  float32  RGB [0,1]
```

**如果预处理不一致会怎样？**  
模型输出会完全错误（得分极低，检测不到任何目标），就像把相机滤镜设置错了——模型看到的和它训练时看到的"世界"不同。

### 7.2 CUDA 异步/同步机制

```cpp
// 异步：把任务加入 GPU 工作队列，CPU 立即返回继续运行
context_->enqueueV3(stream_);

// ── 双引擎并行示例 ────────────────────────────────────────────────
// 时序：
// CPU:  [inferAsync1] [inferAsync2] [syncAndCollect1] [syncAndCollect2]
//        0ms           5ms           20ms              20ms（复用）
// GPU:  [=== engine1 推理 60ms ===]
//                       [=== engine2 推理 60ms ===]
// 总耗时约 80ms，而不是串行的 120ms

detector_->inferAsync(frame);   // GPU 上开始运行 engine1
detector2_->inferAsync(frame);  // GPU 上开始运行 engine2（并发）
auto r1 = detector_->syncAndCollect(frame, thr);   // 等待 engine1 完成
auto r2 = detector2_->syncAndCollect(frame, thr);  // 等待 engine2 完成
// 注：Jetson iGPU 是共享内存架构，实际并行效果有限

// 同步：阻塞 CPU 直到 GPU 这个流上的所有任务完成
cudaStreamSynchronize(stream_);
```

### 7.3 IoU（交并比）计算

IoU 用于判断预测框与 GT 框的重叠程度，是目标检测的核心评估指标：

```python
def iou(a, b):
    # a, b 都是 [x1, y1, x2, y2] 格式
    ix1 = max(a[0], b[0])   # 交叉区域左边界（取较大值）
    iy1 = max(a[1], b[1])   # 交叉区域上边界
    ix2 = min(a[2], b[2])   # 交叉区域右边界（取较小值）
    iy2 = min(a[3], b[3])   # 交叉区域下边界
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)  # 交叉面积
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union

# IoU = 0.0 → 完全不重叠
# IoU = 0.5 → 中等重叠（PASCAL VOC 命中阈值）
# IoU = 1.0 → 完全相同
```

---

## 8. 常用操作命令

### 8.1 编译与运行

```bash
# ── 编译 ────────────────────────────────────────────────────────────
cd /home/nvidia/ros2_ws
colcon build --packages-select faster_rcnn_ros
source install/setup.bash

# ── GPU 锁频（每次重启后需执行一次）────────────────────────────────
sudo jetson_clocks

# ── 话题模式（等待相机输入）────────────────────────────────────────
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml

# ── 文件模式（批量处理图片）────────────────────────────────────────
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
  input_path:=/home/nvidia/ros2_ws/test_images/kitti_100/images \
  output_path:=/home/nvidia/ros2_ws/test_images/results

# ── 文件模式（处理单张图片）────────────────────────────────────────
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
  input_path:=/home/nvidia/ros2_ws/test_images/kitti_100/images/000000.png \
  output_path:=/home/nvidia/ros2_ws/result.jpg

# ── 文件模式（处理视频）────────────────────────────────────────────
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
  input_path:=/home/nvidia/ros2_ws/demo.mp4 \
  output_path:=/home/nvidia/ros2_ws/demo_result.mp4

# ── 覆盖阈值参数 ─────────────────────────────────────────────────
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml threshold:=0.3
```

### 8.2 ROS2 调试命令

```bash
# 查看当前所有活跃话题
ros2 topic list

# 实时打印话题消息（图像话题会打印元信息）
ros2 topic echo /detectnet/overlay --no-arr

# 查看话题频率
ros2 topic hz /detectnet/overlay

# 查看节点信息（发布/订阅的话题列表）
ros2 node info /faster_rcnn_node

# 实时查看节点日志（等同于终端输出）
ros2 node info /faster_rcnn_node

# 发布测试消息（用 test_pub.py 更方便）
python3 /home/nvidia/ros2_ws/test_pub.py
```

### 8.3 模型构建命令

```bash
# 方式1：Python 脚本（含 ONNX 兼容性修复）
cd /home/nvidia/ros2_ws/src/faster_rcnn_ros/models
python3 build_engine.py --height 500 --width 1242
# 输出：faster_rcnn_500.engine（FP16），约需 5-10 分钟

# 方式2：Shell 脚本（构建 FP16 + INT8）
bash build_all_precisions.sh 500 1242
# 输出：faster_rcnn_500_fp16.engine, faster_rcnn_500_int8.engine

# 方式3：直接用 trtexec（适合已有修复好的 ONNX）
trtexec \
  --onnx=fasterrcnn_nuscenes_fixed_500x1242.onnx \
  --fp16 \
  --saveEngine=faster_rcnn_500.engine \
  --shapes=image:1x3x500x1242
```

### 8.4 精度评估命令

```bash
# 4 种配置完整评估（需要 KITTI GT 标注）
python3 /home/nvidia/ros2_ws/gt_eval_all.py

# 结果保存到
cat /home/nvidia/ros2_ws/test_images/compare_results/gt_eval_summary.json
```

---

## 9. 性能评估结果

### 9.1 硬件环境

| 项目 | 规格 |
|------|------|
| 平台 | Jetson AGX Orin Developer Kit |
| GPU | Ampere Architecture iGPU，1024 CUDA Cores |
| GPU 最高频率 | 930 MHz（需 `sudo jetson_clocks` 锁频） |
| 内存 | 32 GB LPDDR5（CPU/GPU 共享） |
| TensorRT | 8.5.2.2 |
| 测试数据集 | KITTI 100 张图片（370×1224 像素） |

### 9.2 四配置综合指标（IoU ≥ 0.5）

| 配置 | 引擎 | GT | Pred | Hit | Miss | FP | **Recall** | **Precision** | 延迟均值 | 延迟P95 |
|------|------|:--:|:----:|:---:|:----:|:--:|:----------:|:-------------:|:--------:|:-------:|
| FP16 + thr=0.5 | 114MB | 429 | 287 | 210 | 219 | 77 | **49.0%** | **73.2%** | 71.4ms | 77.7ms |
| FP16 + thr=0.25 | 114MB | 429 | 980 | 309 | 120 | 671 | **72.0%** | **31.5%** | 70.7ms | 74.1ms |
| INT8 + thr=0.5 | 58MB | 429 | 272 | 205 | 224 | 67 | **47.8%** | **75.4%** | 62.9ms | 65.7ms |
| INT8 + thr=0.25 | 58MB | 429 | 1093 | 314 | 115 | 779 | **73.2%** | **28.7%** | 63.1ms | 68.3ms |

**指标说明：**
- **GT**：测试集中的真实目标总数
- **Pred**：模型预测的检测框总数
- **Hit**：预测框与 GT 框 IoU≥0.5 且类别正确（真正例 TP）
- **Miss**：GT 框没有被任何预测框命中（漏检，假负例 FN）
- **FP**：预测框没有匹配到任何 GT 框（误检，假正例）
- **Recall（召回率）**= Hit / GT × 100%
- **Precision（精确率）**= Hit / Pred × 100%

### 9.3 关键对比结论

```
┌─────────────────────────────────────────────────────────────────┐
│  阈值 0.5 → 0.25（FP16）                                        │
│  Recall↑ +23.1%（49%→72%）  但  FP↑ +594（77→671）            │
│  → 找到更多目标，但误检大幅增加                                  │
├─────────────────────────────────────────────────────────────────┤
│  FP16 → INT8（阈值=0.5）★ 推荐方案                              │
│  Recall 仅 -1.2%（精度几乎无损）                                 │
│  延迟快 8.5ms，引擎减小 56MB（节省一半存储）                     │
├─────────────────────────────────────────────────────────────────┤
│  为什么 Faster R-CNN 不能达到 <10ms？                            │
│  两阶段架构：Stage1（RPN）→ Stage2（ROI Head）串行执行           │
│  无法像 YOLO 的单阶段检测器那样在 3ms 内完成                     │
│  YOLO v8n 在相同硬件上约 3.1ms（单阶段，精度较低）              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 常见问题排查

### Q1：编译失败，找不到 `nvinfer.h`

```bash
# 检查 TensorRT 是否安装
dpkg -l | grep tensorrt
ls /usr/include/NvInfer.h  # 头文件应存在

# 如果缺少，在 CMakeLists.txt 中添加搜索路径
include_directories(/usr/include)
```

### Q2：启动节点后立即崩溃

```bash
# 查看详细错误
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml 2>&1

# 常见原因：
# 1. 引擎文件路径错误 → 检查 engine_path 参数
ls install/faster_rcnn_ros/share/faster_rcnn_ros/models/
# 2. 引擎与当前 GPU 不匹配（在不同机器构建的引擎无法跨机使用）
# 3. 显存不足 → 检查 sudo tegrastats
```

### Q3：检测框全为 0，没有任何检测结果

```bash
# 原因1：阈值太高（改为 0.1 测试）
ros2 launch ... threshold:=0.1

# 原因2：GPU 频率太低
sudo tegrastats | grep GR3D_FREQ  # 应为 930MHz
sudo jetson_clocks  # 锁频

# 原因3：图像话题编码格式不匹配
ros2 topic echo /detectnet/image_in | grep encoding  # 应为 "bgr8" 或 "rgb8"
```

### Q4：推理延迟突然从 70ms 变为 300ms+

```bash
# 通常是 GPU 降频（温度过高或未锁频）
sudo tegrastats
# GR3D_FREQ 应为 930MHz，若为 306MHz 则是降频

sudo jetson_clocks
```

### Q5：`source install/setup.bash` 后仍找不到包

```bash
# 确认编译成功
colcon build --packages-select faster_rcnn_ros
echo $?  # 应为 0

# 确认 source 了正确环境
echo $AMENT_PREFIX_PATH  # 应包含 /home/nvidia/ros2_ws/install

# 重新 source
source /home/nvidia/ros2_ws/install/setup.bash
```

### Q6：如何给新数据集训练？

参考 `src/faster_rcnn_ros/training/` 目录：

```bash
# 1. 准备数据集（NuScenes 格式）
# 参考 nuscenes_dataset.py 实现自定义数据集

# 2. 训练
python3 training/train.py \
  --data_root /path/to/dataset \
  --epochs 30 \
  --batch_size 4

# 3. 导出 ONNX
python3 training/export_onnx.py \
  --checkpoint best_model.pth \
  --output model.onnx

# 4. 构建 TRT 引擎
python3 src/faster_rcnn_ros/models/build_engine.py \
  --onnx model.onnx \
  --height 500 --width 1242
```

---

## 11. ROS2 对多传感器系统的优势与必要性

### 11.1 当前项目中 ROS2 解决的核心问题

传统方案（纯 C++ 程序 + OpenCV）能做图像推理，但无法应对多传感器系统。ROS2 的价值体现在：

| 维度 | 没有 ROS2 的困境 | ROS2 的解法 |
|------|----------------|-------------|
| **传感器接入** | 每个传感器需自己写驱动、协议解析代码 | 厂商直接提供 ROS2 驱动包，即插即用 |
| **进程解耦** | 推理程序崩溃 = 整个系统崩溃 | 每个节点独立进程，检测器挂了不影响相机采集 |
| **时间同步** | 手动处理多传感器时间戳对齐极复杂 | `message_filters::TimeSynchronizer` 内置同步 |
| **数据记录** | 需自己写录制/回放工具 | `ros2 bag record` 一行命令录制所有话题 |
| **可视化** | 自己写 OpenCV 窗口显示 | RViz2 直接显示点云/图像/检测框 |

### 11.2 后续接入毫米波雷达的价值（最关键）

毫米波雷达数据处理是 ROS2 价值体现最明显的场景：

```
当前系统（纯相机）：
  相机 → [faster_rcnn_node] → /detectnet/overlay

未来융合系统（相机 + 毫米波雷达）：
  相机       ─────────────────────────▶ /detectnet/overlay ─┐
  毫米波雷达 → [radar_driver_node] ──▶ /radar/targets ──────┤
                                                              ▼
                                         [fusion_node]（目标级融合）
                                              │
                                              ▼
                                     /fusion/detections（最终结果）
```

**具体好处：**

**① 雷达驱动零开发成本**

主流毫米波雷达厂商（Continental ARS408、TI AWR1843 等）都提供 ROS2 驱动包，直接安装即用，不需要自己解析 CAN 总线协议。

**② 时间戳自动对齐**

相机 30fps（约 33ms 周期）、雷达 20Hz（50ms 周期），两者触发时刻天然不同步。ROS2 的 `ApproximateTimeSynchronizer` 会自动寻找时间最近的帧配对，误差可控制在 10ms 以内，不需要自己写缓冲队列。

```cpp
// 融合节点中的时间同步示例（仅需几行代码）
message_filters::Subscriber<Image> img_sub(this, "/detectnet/image_in");
message_filters::Subscriber<RadarScan> radar_sub(this, "/radar/targets");
typedef sync_policies::ApproximateTime<Image, RadarScan> Policy;
Synchronizer<Policy> sync(Policy(10), img_sub, radar_sub);
sync.registerCallback(&FusionNode::fusionCallback, this);
```

**③ 融合节点与检测节点完全独立**

融合算法修改后不需要重新编译检测节点：
```bash
# 只重新编译融合节点
colcon build --packages-select radar_fusion_ros
# 检测节点无需重新编译，直接重启即可
```

**④ 标准消息格式，无私有协议**

| 传感器 | ROS2 消息类型 | 字段含义 |
|--------|--------------|----------|
| 相机 | `sensor_msgs/Image` | height/width/encoding/data |
| 毫米波雷达 | `sensor_msgs/PointCloud2` | x/y/z/velocity/intensity |
| 激光雷达 | `sensor_msgs/PointCloud2` | 与雷达相同格式，无缝融合 |
| GPS/IMU | `sensor_msgs/NavSatFix` | latitude/longitude/altitude |

**⑤ `ros2 bag` 数据录制——调试效率倍增**

调试融合算法时，录制一次真实场景的全部传感器数据，之后可无限次回放，不需要每次都开车测试：

```bash
# 录制所有传感器数据到 bag 文件
ros2 bag record /detectnet/image_in /radar/targets /imu/data

# 回放（完全模拟真实传感器，融合节点感知不到差异）
ros2 bag play dataset_20260318.bag

# 查看 bag 文件内容
ros2 bag info dataset_20260318.bag
```

### 11.3 扩展路线图

```
当前：相机 → Faster R-CNN → 2D 检测框
                    ↓
第一步（融合）：+ 毫米波雷达 → 2D 框 + 速度/距离信息
                    ↓
第二步（3D感知）：+ 激光雷达 → 3D 检测框（使用 PointPillars 等）
                    ↓
第三步（全栈感知）：+ GPS/IMU → 目标跟踪 + 位置预测

每一步都只需要新增 ROS2 节点，已有节点不需要修改
```

---

## 12. 单引擎与双引擎模式详解

### 12.1 当前运行的是单引擎模式

查看 launch 文件配置：

```xml
<!-- launch/faster_rcnn.launch.xml -->
<arg name="engine_path_2" default="none"/>  <!-- none = 禁用第二引擎 -->
```

注释说明：**当前已禁用，单 500×1242 引擎即可达到 ≥80% Recall，无需双引擎。**

当 `engine_path_2="none"` 时，节点代码第 106 行的条件判断：

```cpp
// faster_rcnn_node.cpp
if (!engine_path_2.empty() && engine_path_2 != "none") {
    detector2_ = ...;   // 不会执行，detector2_ 保持 nullptr
}
```

进而在 `runInference()` 中走**快速路径**：

```cpp
if (!detector2_ && ped_threshold_ >= threshold_) {
    return detector_->infer(frame, threshold_);  // ← 当前默认走这里
}
```

### 12.2 三条执行路径完整解析

```
runInference() 调用时，根据配置选择路径：

路径 A（当前默认）：detector2_=null，ped_thr >= car_thr
    └─→ 快速路径：detector_->infer()
        单次同步推理，代码最简，延迟最低

路径 B：detector2_=null，但 ped_thr < car_thr（类别专属阈值）
    └─→ 先用最低阈值捞出所有框
        → inferAsync() → syncAndCollect() → filterAndNMS()
        （同一引擎，但对行人/车辆使用不同阈值精细过滤）

路径 C（双引擎模式）：detector2_ 非空
    ├─→ detector_->inferAsync(frame)   ← 引擎1 入队 GPU stream1（不等待）
    ├─→ detector2_->inferAsync(frame)  ← 引擎2 立即入队 GPU stream2（不等待）
    ├─→ detector_->syncAndCollect()    ← 等待引擎1 完成，取结果
    ├─→ detector2_->syncAndCollect()   ← 等待引擎2 完成，取结果
    ├─→ 合并两组结果
    └─→ filterAndNMS（类别专属阈值 + NMS 去重）
```

### 12.3 双引擎的设计意图与为何默认禁用

**设计意图**：多尺度检测。

- 引擎1（500×1242）：适合检测中远距离目标（目标在图像中较小）
- 引擎2（不同分辨率）：适合检测近距离大目标（需要不同感受野）
- 合并两引擎结果 → 提升全距离召回率

**为何默认禁用**：

1. **单引擎已够用**：单 500×1242 引擎 Recall=81.8%（IoU≥0.5），满足部署需求
2. **Jetson iGPU 无真并行**：Jetson AGX Orin 的 CPU 和 GPU 共享内存，两个引擎实际上在同一 GPU 上串行执行，无法真正并行加速
3. **显存占用翻倍**：两个引擎同时驻留，显存压力增大

### 12.4 如何手动开启双引擎（仅供参考）

```bash
# 使用 INT8 引擎作为第二引擎（FP16 检测车辆，INT8 专注行人低阈值）
ros2 launch faster_rcnn_ros faster_rcnn.launch.xml \
  engine_path_2:=$(ros2 pkg prefix faster_rcnn_ros)/share/faster_rcnn_ros/models/faster_rcnn_500_int8.engine \
  ped_threshold:=0.3
  # ↑ 行人用 0.3，比车辆阈值 0.5 低，提升行人召回率
```

---

## 13. 培训重点讲解指南

本章按**讲解优先级**排序，帮助授课人把握重点。

### 13.1 ★★★ 最高优先：预处理一致性（对应第 7.1 节）

这是最容易出错、最难排查的问题，必须反复强调。

```cpp
cv::dnn::blobFromImage(image, blob_,
    1.0/255.0,           // ① 归一化系数（必须与训练一致）
    cv::Size(1242, 500), // ② 缩放尺寸（宽在前！）
    cv::Scalar(),        // ③ 不减均值
    /*swapRB=*/true,     // ④ BGR→RGB（最容易遗漏！）
    /*crop=*/false);     // ⑤ 全图缩放，不裁剪
```

**重点强调**：这 5 个参数必须与模型训练时的预处理完全一致。

任何一个不对（**尤其是 `swapRB`**），模型输出全为噪声，所有检测框置信度 < 0.1，表现为「完全检测不到任何目标」——**程序不报任何错误**，这让 Bug 极难定位。

**演示建议**：现场将 `swapRB` 改为 `false`，重新推理，让学员亲眼看到检测结果消失。

### 13.2 ★★★ 最高优先：TRT 引擎平台绑定性（对应第 5.3 节）

```
.engine 文件 ≠ 通用的模型文件
它是以下三个因素的乘积，缺一不可：

  当前 GPU 型号  ×  当前 TRT 版本  ×  当前输入分辨率
  ─────────────────────────────────────────────────────
  Jetson AGX Orin 构建  →  不能在 TX2 / PC 上运行
  TRT 8.5 构建          →  不能在 TRT 7.x / 8.6 上运行
  500×1242 构建         →  输入必须是 500×1242，不能动态改
  构建时间约 5-10 分钟  →  必须在目标机器上重新构建
```

**这是学员最常产生困惑的点**：「为什么把引擎文件拷贝到另一台机器就不能用？」

### 13.3 ★★ 重要：动态输出形状（`-1` 输出的含义）（对应第 5.4 节）

```
模型输出：scores[-1]  labels[-1]  boxes[-1,4]
                 ↑
    这个 -1 的意思：每张图片检测到的目标数不固定
    图像全黑可能是 0 个目标
    拥挤场景可能是 300 个目标
    TRT 在每次推理前动态分配 GPU 显存
```

要结合代码讲：`OutputAllocator::reallocateOutput()` 会在每次推理前被 TRT 自动调用分配显存，`notifyShape()` 告知实际输出了多少个框。这是 TRT 8.5 新引入的机制，与 TRT 7.x 的静态输出分配有根本区别。

### 13.4 ★★ 重要：数据流全流程（对应第 6.1 节）

建议对照流程图逐步演示：

1. 用 `ros2 launch` 启动，看到 `[文件模式]` 或 `[话题模式]` 日志
2. 触发一次推理，对照时序图讲 `inferAsync` → `cudaStreamSynchronize` → 坐标缩放
3. 看到 `[000049.png] 11 个目标 | 84.8 ms` 日志，解释每个字段含义

**核心 insight**：从 `cv::imread` 到得到检测框，经历了 **4 次内存搬运**：

```
CPU: imread → blobFromImage（缩放+归一化，~1ms）
                    ↓ H2D cudaMemcpyAsync（~0.5ms）
GPU:          TRT 推理（~63-71ms，延迟主体）
                    ↓ D2H cudaMemcpy（~0.1ms）
CPU:          坐标缩放 + 阈值过滤（~0.1ms）
```

结论：**延迟主要来自 GPU 推理本身，不是数据传输**。

### 13.5 ★ 次要：精度 vs 速度权衡（对应第 9.3 节）

对照四配置对比表，引导学员理解没有完美配置，只有根据场景的权衡：

```
                    Precision（精确率）
                         高 ↑
                 INT8+0.5 ●    ● FP16+0.5


                 INT8+0.25 ●   ● FP16+0.25
                         低 ↓
               低 ←  Recall（召回率）  → 高
```

**选择原则：**
- 误检代价高（如自动紧急制动，误判=误刹车）→ 选 `thr=0.5`（高精确率）
- 漏检代价高（如行人检测，漏检=事故）→ 选 `thr=0.25`（高召回，但接受误检）
- 资源受限/嵌入式设备 → 选 INT8（引擎只有 58MB，延迟快 8ms）

### 13.6 培训课程建议时间分配

| 内容 | 建议时长 | 方式 |
|------|---------|------|
| ROS2 概念快速入门（第1章） | 15 分钟 | 幻灯片讲解 |
| 系统架构图讲解（第2章） | 10 分钟 | 白板画图 |
| `demo.sh` 现场演示 | 10 分钟 | 终端实操 |
| 预处理一致性（★★★ 13.1） | 10 分钟 | 代码演示 + 故意制造 Bug |
| TRT 引擎绑定性（★★★ 13.2） | 10 分钟 | 讲解 + 问答 |
| 数据流全流程（★★ 13.4） | 10 分钟 | 对照流程图 |
| 精度对比分析（★ 13.5） | 10 分钟 | 对照表格 |
| Q&A + 上手实验 | 25 分钟 | 自由操作 |
| **合计** | **100 分钟** | |

---

## 附录：快速参考卡

```
┌──────────────────────────────────────────────────────────────────────┐
│                    快速参考卡                                         │
├────────────────┬─────────────────────────────────────────────────────┤
│ 编译           │ cd ros2_ws && colcon build --packages-select faster_ │
│                │ rcnn_ros && source install/setup.bash               │
├────────────────┼─────────────────────────────────────────────────────┤
│ GPU 锁频       │ sudo jetson_clocks                                   │
├────────────────┼─────────────────────────────────────────────────────┤
│ 批量推理       │ ros2 launch faster_rcnn_ros faster_rcnn.launch.xml  │
│                │   input_path:=<图片目录> output_path:=<输出目录>     │
├────────────────┼─────────────────────────────────────────────────────┤
│ 精度评估       │ python3 gt_eval_all.py                               │
├────────────────┼─────────────────────────────────────────────────────┤
│ 引擎构建       │ python3 src/faster_rcnn_ros/models/build_engine.py   │
│                │   --height 500 --width 1242                         │
├────────────────┼─────────────────────────────────────────────────────┤
│ 查看话题       │ ros2 topic list / ros2 topic hz /detectnet/overlay   │
├────────────────┼─────────────────────────────────────────────────────┤
│ 推荐配置       │ INT8 引擎 + threshold=0.5（精度无损，延迟最快）       │
├────────────────┼─────────────────────────────────────────────────────┤
│ 引擎路径       │ install/faster_rcnn_ros/share/faster_rcnn_ros/models/│
└────────────────┴─────────────────────────────────────────────────────┘
```
