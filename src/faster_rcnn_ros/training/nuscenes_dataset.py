"""
nuscenes_dataset.py
NuScenes 数据集加载器，适配 torchvision FasterRCNN 训练接口。

依赖:
  pip install nuscenes-devkit

NuScenes 类别 → 模型类别 ID 映射（num_classes=10）:
  background=0, car=1, truck=2, bus=3, trailer=4,
  construction_vehicle=5, pedestrian=6, motorcycle=7, bicycle=8, traffic_cone=9

用法:
  dataset = NuScenesDataset(
      dataroot='/path/to/nuscenes',
      version='v1.0-mini',         # 或 'v1.0-trainval'
      camera='CAM_FRONT',          # 前视相机
      transforms=get_train_transforms(),
  )
"""
import os
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# NuScenes 类别名 → 模型 label id
# 参考 nuScenes 官方 detection 类别定义
NUSCENES_CATEGORY_MAP: Dict[str, int] = {
    "car":                   1,
    "truck":                 2,
    "bus":                   3,
    "trailer":               4,
    "construction_vehicle":  5,
    "pedestrian":            6,
    "motorcycle":            7,
    "bicycle":               8,
    "traffic_cone":          9,
    # 以下类别不在模型中，跳过
    "barrier":               0,   # 映射为 background，训练时过滤
}

# NuScenes devkit 的类别名称 → 我们的类别名（合并同义词）
NUSC_TO_OUR: Dict[str, str] = {
    "vehicle.car":                      "car",
    "vehicle.truck":                    "truck",
    "vehicle.bus.bendy":                "bus",
    "vehicle.bus.rigid":                "bus",
    "vehicle.trailer":                  "trailer",
    "vehicle.construction":             "construction_vehicle",
    "human.pedestrian.adult":           "pedestrian",
    "human.pedestrian.child":           "pedestrian",
    "human.pedestrian.wheelchair":      "pedestrian",
    "human.pedestrian.stroller":        "pedestrian",
    "human.pedestrian.personal_mobility":"pedestrian",
    "human.pedestrian.police_officer":  "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "vehicle.motorcycle":               "motorcycle",
    "vehicle.bicycle":                  "bicycle",
    "movable_object.trafficcone":       "traffic_cone",
    "movable_object.barrier":           "barrier",
    "movable_object.debris":            "barrier",
    "movable_object.pushable_pullable": "barrier",
    "static_object.bicycle_rack":       "bicycle",
    "animal":                           "pedestrian",
}

# 支持的相机通道
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def _nusc_category_to_label(nusc_cat: str) -> Optional[int]:
    """将 NuScenes 原始类别名转换为模型 label id，无法映射返回 None。"""
    our_cat = NUSC_TO_OUR.get(nusc_cat)
    if our_cat is None:
        return None
    label = NUSCENES_CATEGORY_MAP.get(our_cat, 0)
    return label if label > 0 else None   # 过滤 background


class NuScenesDataset(Dataset):
    """
    NuScenes 2D 检测数据集。
    使用 nuscenes-devkit 将 3D box 投影到指定相机的图像平面，
    生成 2D bounding box 用于 Faster RCNN 训练。

    :param dataroot:   NuScenes 数据集根目录（含 v1.0-xxx/）
    :param version:    数据集版本，'v1.0-mini' / 'v1.0-trainval'
    :param split:      'train' 或 'val'（对 mini 版本均使用全部场景）
    :param cameras:    使用的相机列表（默认仅前视 CAM_FRONT）
    :param transforms: 数据增强（接收 image, target 两个参数）
    :param min_visibility: 可见度过滤（1-4，4=完全可见）
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "train",
        cameras: Optional[List[str]] = None,
        transforms=None,
        min_visibility: int = 1,
    ):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes

        self.dataroot = dataroot
        self.transforms = transforms
        self.min_visibility = min_visibility

        if cameras is None:
            cameras = ["CAM_FRONT"]
        self.cameras = cameras

        print(f"  加载 NuScenes {version} ({split})...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # 按 split 筛选 scene
        if version == "v1.0-mini":
            scene_names = [s["name"] for s in self.nusc.scene]
        else:
            splits = create_splits_scenes()
            scene_names = splits[split]

        scene_tokens = {s["name"]: s["token"] for s in self.nusc.scene}
        valid_tokens = {scene_tokens[n] for n in scene_names if n in scene_tokens}

        # 收集所有 sample + camera 组合
        self.samples: List[Tuple[str, str]] = []   # (sample_token, camera)
        for sample in self.nusc.sample:
            if sample["scene_token"] in valid_tokens:
                for cam in self.cameras:
                    if cam in sample["data"]:
                        self.samples.append((sample["token"], cam))

        print(f"  样本数量: {len(self.samples)} (场景数: {len(valid_tokens)}, 相机: {cameras})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility

        sample_token, cam = self.samples[idx]
        sample = self.nusc.get("sample", sample_token)
        sd_token = sample["data"][cam]
        sd = self.nusc.get("sample_data", sd_token)

        # 加载图像
        img_path = os.path.join(self.dataroot, sd["filename"])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # 获取相机内参和位姿
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        import pyquaternion
        from nuscenes.utils.data_classes import Box
        import numpy as np

        K = np.array(cs["camera_intrinsic"])

        # 获取当前 sample 的所有 3D box
        boxes, vis_tokens = self.nusc.get_sample_data(
            sd_token,
            box_vis_level=BoxVisibility.ANY,
        )

        gt_boxes: List[List[float]] = []
        gt_labels: List[int] = []

        for box in boxes:
            # 可见度过滤（NuScenes 中每个 annotation 有 visibility_token）
            ann_token = box.token
            try:
                ann = self.nusc.get("sample_annotation", ann_token)
                vis_token = ann.get("visibility_token", "4")
                vis_level = int(self.nusc.get("visibility", vis_token)["level"].split("-")[0]) \
                    if vis_token else 4
                if vis_level < self.min_visibility:
                    continue
            except Exception:
                pass

            # 类别映射
            label = _nusc_category_to_label(box.name)
            if label is None:
                continue

            # 将 3D box 投影为 2D bbox
            corners = box.corners()               # (3, 8)
            # 投影到像素坐标
            corners_h = np.vstack([corners, np.ones((1, 8))])
            corners_img = K @ corners_h[:3]       # (3, 8)
            corners_img /= corners_img[2:3]       # 归一化
            xs = corners_img[0]
            ys = corners_img[1]

            x1 = float(np.clip(xs.min(), 0, W - 1))
            y1 = float(np.clip(ys.min(), 0, H - 1))
            x2 = float(np.clip(xs.max(), 0, W - 1))
            y2 = float(np.clip(ys.max(), 0, H - 1))

            # 过滤退化框（面积过小）
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(label)

        # 转为 tensor
        if len(gt_boxes) > 0:
            boxes_t  = torch.tensor(gt_boxes,  dtype=torch.float32)
            labels_t = torch.tensor(gt_labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":     boxes_t,
            "labels":    labels_t,
            "image_id":  torch.tensor([idx]),
        }

        # PIL → Tensor [0,1]
        import torchvision.transforms.functional as TF
        image_t = TF.to_tensor(image)

        if self.transforms is not None:
            image_t, target = self.transforms(image_t, target)

        return image_t, target


def collate_fn(batch):
    return tuple(zip(*batch))
