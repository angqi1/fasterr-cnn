"""
transforms.py
用于 Faster RCNN 训练的数据增强，同时变换图像和 bounding box。

增强策略（训练集）:
  1. 随机水平翻转
  2. 随机光度扰动（亮度/对比度/饱和度/色调）
  3. 随机缩放（0.5x ~ 1.5x）
  4. 随机裁剪（保留至少 70% 面积有效的 box）

验证集只做 Resize，不做随机增强。
"""
import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, H, W = image.shape
            image = TF.hflip(image)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomPhotometricDistort:
    """随机亮度/对比度/饱和度/色调扰动，增强不同光照条件下的泛化能力。"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
        self.jitter = ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.jitter(image)
        return image, target


class RandomScale:
    """
    随机缩放整张图像（包括 box 坐标同步缩放）。
    scale_range: 缩放比例范围，如 (0.5, 1.5)。
    """
    def __init__(self, scale_range=(0.7, 1.3), prob: float = 0.5):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        scale = random.uniform(*self.scale_range)
        _, H, W = image.shape
        new_H = max(int(H * scale), 1)
        new_W = max(int(W * scale), 1)

        image = TF.resize(image, [new_H, new_W])

        if target["boxes"].numel() > 0:
            target["boxes"] = target["boxes"] * scale

        return image, target


class RandomCrop:
    """
    随机裁剪，保留中心区域为 crop_ratio 比例。
    裁剪后过滤掉面积明显缩小的 box（保留 keep_ratio 以上的 box）。
    """
    def __init__(self, crop_ratio: float = 0.8, keep_ratio: float = 0.5, prob: float = 0.3):
        self.crop_ratio = crop_ratio
        self.keep_ratio = keep_ratio
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        _, H, W = image.shape
        new_H = int(H * self.crop_ratio)
        new_W = int(W * self.crop_ratio)
        top    = random.randint(0, H - new_H)
        left   = random.randint(0, W - new_W)

        image = TF.crop(image, top, left, new_H, new_W)

        if target["boxes"].numel() > 0:
            boxes  = target["boxes"].clone()
            labels = target["labels"].clone()

            # 将 box 坐标转换到裁剪后的坐标系
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            boxes = boxes.clamp(
                min=0,
                max=torch.tensor([new_W, new_H, new_W, new_H], dtype=torch.float32)
            )

            # 计算裁剪后面积 / 原面积，过滤面积损失过大的 box
            orig_area = (target["boxes"][:, 2] - target["boxes"][:, 0]) * \
                        (target["boxes"][:, 3] - target["boxes"][:, 1])
            new_area  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep_mask = (new_area / orig_area.clamp(min=1e-6)) >= self.keep_ratio
            keep_mask &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

            target["boxes"]  = boxes[keep_mask]
            target["labels"] = labels[keep_mask]

        return image, target


class ResizeTo:
    """将图像缩放到固定大小（推理输入尺寸），box 坐标同步缩放。"""
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width

    def __call__(self, image, target):
        _, H, W = image.shape
        image = TF.resize(image, [self.h, self.w])
        if target["boxes"].numel() > 0:
            sx = self.w / W
            sy = self.h / H
            scale = torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
            target["boxes"] = target["boxes"] * scale
        return image, target


def get_train_transforms(input_h: int = 375, input_w: int = 1242):
    """
    训练数据增强流程：
      随机翻转 → 光度扰动 → 随机缩放 → 随机裁剪 → Resize 到网络输入尺寸
    """
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        RandomPhotometricDistort(prob=0.5),
        RandomScale(scale_range=(0.7, 1.3), prob=0.5),
        RandomCrop(crop_ratio=0.85, keep_ratio=0.5, prob=0.3),
        ResizeTo(input_h, input_w),
    ])


def get_val_transforms(input_h: int = 375, input_w: int = 1242):
    """验证集只做 Resize，不引入随机性。"""
    return Compose([
        ResizeTo(input_h, input_w),
    ])
