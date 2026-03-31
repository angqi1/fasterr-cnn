# 数据集使用清单

> 本文件记录项目中使用的所有数据集，刷机重装后可按此指引重新获取。
> 数据集文件体积较大（总计约 12GB），不纳入 git 管理，需手动重新下载。

---

## 1. SSLAD-2D（主要评估数据集）

### 用途
项目核心测试集。用于 Faster R-CNN vs YOLOv8 对比评估，以及硬件优化前后的推理性能测试。

### 来源
- **官网**：https://sslad2021.github.io/pages/challenge.html
- **下载地址**：https://sslad2021.github.io/pages/challenge.html（需登录注册）
- 下载文件：`labeled_trainval.tar`（3.0GB）、`labeled_test.tar`（2.7GB）

### 解压方式
```bash
# 解压到 test_images/ 
cd /home/nvidia/ros2_ws/test_images
tar -xf labeled_trainval.tar    # 解压后得到 SSLAD-2D/
tar -xf labeled_test.tar
```

### 目录结构（解压后）
```
test_images/SSLAD-2D/
    annotations/
        instance_val.json       # COCO 格式标注
    images/
        val/                    # 验证集图片（1920×1080 JPG）
```

### sslad_300 子集（300张评估集）

项目中使用 **`prepare_sslad_300.py`** 从 val 集中每隔 N 帧均匀采样 300 张构建测试子集，标注转换为 KITTI 格式 txt。

**图片文件列表保存在** `test_images/sslad_300/image_list.txt`（共 300 条）。

重新准备方式：
```bash
# SSLAD-2D 解压完成后执行
python3 prepare_sslad_300.py
```

### 类别映射（SSLAD-2D → KITTI 格式）
| SSLAD 类别 | KITTI 类别名 |
|---|---|
| 1: Pedestrian | Pedestrian |
| 2: Cyclist | Cyclist |
| 3: Car | Car |
| 4: Truck | Truck |
| 5: Tram | Tram |
| 6: Tricycle | Cyclist（归并）|

### 统计信息（sslad_300 子集）
| 类别 | 目标数量 |
|---|---|
| Car | 1233 |
| Truck | 463 |
| Cyclist | 178 |
| Pedestrian | 141 |
| Tram | 82 |
| **总计** | **2097** |

---

## 2. nuImages（辅助验证数据集）

### 用途
用于 Faster R-CNN nuScenes 训练模型的原始域内精度验证，与 SSLAD 做跨域对比。

### 来源
- **官网**：https://www.nuscenes.org/nuimages
- 下载需注册账号
- 下载脚本：`download_nuimages.py`、`download_nuimages_uniform.py`

### 重新下载
```bash
python3 download_nuimages.py           # 随机采样版
python3 download_nuimages_uniform.py   # 均匀采样版
```

### 目录
```
test_images/nuimages/          # 随机采样，约 74MB
test_images/nuimages_uniform/  # 均匀采样，约 52KB（仅含列表）
```

---

## 3. nuScenes mini（辅助验证）

### 用途
nuScenes 传感器融合场景测试，含摄像头+雷达数据。

### 来源
- **官网**：https://www.nuscenes.org/nuscenes#download
- 下载脚本：`download_nuscenes_mini.py`

### 重新下载
```bash
python3 download_nuscenes_mini.py
```

### 目录
```
test_images/nuscenes/    # 约 229MB
```

---

## 4. KITTI（早期验证数据集）

### 用途
项目早期使用 KITTI 100 张和 500 张图片进行基础功能验证和性能基准测试（含 INT8 校准）。

### 来源
- **官网**：http://www.cvlibs.net/datasets/kitti/
- 下载脚本：`download_kitti_subset.py`

### 重新下载
```bash
python3 download_kitti_subset.py       # 自动下载 100 张
```

### 目录
```
test_images/kitti_100/   # 100 张
test_images/kitti_500/   # 500 张
```

---

## 5. YOLOv8 预训练权重

### 文件
`yolov8n.pt`（6.3MB，已纳入 git 管理）

### 来源
```bash
pip install ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# 或直接下载
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## 快速恢复脚本

刷机重装后，按以下顺序恢复数据：

```bash
# 1. 克隆代码库
git clone git@github.com:angqi1/fasterr-cnn.git ros2_ws
cd ros2_ws

# 2. 手动将 SSLAD-2D tar 包拷贝到 test_images/（需从其他设备传输）
#    或重新从官网下载后解压

# 3. 重建 sslad_300 子集
python3 prepare_sslad_300.py

# 4. 重新构建 TRT 引擎（引擎不进 git，需在目标设备上重建）
#    见 PIPELINE.md 中的引擎构建步骤

# 5. 重新编译 ROS2 工作空间
source /opt/ros/humble/setup.bash   # JetPack6.x 用 humble
colcon build --symlink-install
```
