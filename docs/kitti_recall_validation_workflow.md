# KITTI 图片推理 Recall 验证完整流程

## 1. 目标

使用 KITTI Object Detection 官方数据集下载至少 100 张图片，执行模型推理并统计 Recall，产出可复现的验证流程与结果。

本次实测已完成：
- KITTI 图片: 100 张
- KITTI 标注: 100 份
- 评估脚本: compare_gt.py（已支持命令行参数）
- 可视化结果: 每张图生成 GT/Pred 对比图

## 2. 数据来源

- 官方图片压缩包:
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
- 官方标签压缩包:
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

说明：
- 采用远程 ZIP 按需解压，只下载子集，不下载完整 12GB 图片包。
- 下载脚本: download_kitti_subset.py

## 3. 目录结构

执行后会生成以下目录：

~~~text
/home/nvidia/ros2_ws/
├── download_kitti_subset.py
├── compare_gt.py
└── test_images/
    └── kitti_100/
        ├── images/          # 100 张 KITTI 图片
        ├── labels/          # 100 个 KITTI 标签 txt
        ├── results_vis/     # 可视化对比图 cmp_*.png
        └── metrics_compare_gt.txt
~~~

## 4. 一键下载 100 张 KITTI 样本

在工作区执行：

~~~bash
cd /home/nvidia/ros2_ws
/usr/bin/python download_kitti_subset.py --count 100 --out-root test_images/kitti_100
~~~

下载完成后检查数量：

~~~bash
ls test_images/kitti_100/images/*.png | wc -l
ls test_images/kitti_100/labels/*.txt | wc -l
~~~

预期输出：
- images: 100
- labels: 100

## 5. 运行 Recall 验证

执行评估：

~~~bash
cd /home/nvidia/ros2_ws
/usr/bin/python compare_gt.py \
  --img-dir test_images/kitti_100/images \
  --gt-dir test_images/kitti_100/labels \
  --out-dir test_images/kitti_100/results_vis \
  --threshold 0.25 \
  --iou-thresh 0.3 \
  | tee test_images/kitti_100/metrics_compare_gt.txt
~~~

脚本输出：
- 每张图的 GT/Pred/Hit/Miss/FP/Recall
- 总体 Recall、Precision、F1
- 按类别 Recall
- 对比可视化图片 cmp_*.png

## 6. 本次实测结果（100 张）

参数：
- threshold = 0.25
- IoU = 0.3
- ONNX = src/faster_rcnn_ros/models/fasterrcnn_nuscenes.onnx

汇总：
- GT 总数: 435
- Pred 总数: 805
- Hit: 341
- Miss: 94
- FP: 464
- Recall: 78.4%
- Precision: 42.4%
- F1-score: 55.0%

按类别 Recall（节选）：
- car: 84.8%
- pedestrian: 76.1%
- van: 66.7%
- cyclist: 47.1%
- tram: 8.3%

## 7. 参数建议

- 如果目标是提高召回：可尝试降低 threshold（例如 0.20）
- 如果目标是降低误检：可提高 threshold（例如 0.30）
- 建议固定 IoU=0.3 做横向对比，再单独做 IoU=0.5 的严格评估

## 8. 复现实验模板

### 8.1 换 200 张样本

~~~bash
/usr/bin/python download_kitti_subset.py --count 200 --out-root test_images/kitti_200
~~~

### 8.2 使用不同阈值评估

~~~bash
/usr/bin/python compare_gt.py \
  --img-dir test_images/kitti_100/images \
  --gt-dir test_images/kitti_100/labels \
  --out-dir test_images/kitti_100/results_t020 \
  --threshold 0.20 --iou-thresh 0.3
~~~

## 9. 注意事项

- compare_gt.py 当前是 IoU 命中驱动的匹配逻辑（类别宽容），适合快速看召回趋势。
- 若要做严格 mAP/按类 AP，请引入 KITTI 官方评估协议或 COCO 风格评估脚本。
- 若使用 TensorRT 节点评估线上延迟，建议固定同一批图片并记录 warmup 后平均值。
