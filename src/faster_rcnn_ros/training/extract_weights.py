"""
extract_weights.py
构建可微调的 PyTorch Faster RCNN 模型，策略：

  ① 加载 COCO 预训练权重（backbone + FPN + RPN 全部继承）
  ② 将检测头替换为 NuScenes 10 类版本
  ③ 保存为 .pth，供 train.py 微调

原始 NuScenes ONNX 因 FrozenBatchNorm2d 常量折叠无法直接还原 PyTorch 模型，
使用 COCO 预训练起点迁移学习效果更好（COCO 含 car/truck/bus/person/
motorcycle/bicycle，与 NuScenes 类别高度重叠）。

用法:
  python3 extract_weights.py --output fasterrcnn_nuscenes_init.pth

  # 可选：指定 COCO checkpoint 路径（默认使用 torch hub 缓存）
  python3 extract_weights.py \
      --coco_ckpt ~/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth \
      --output fasterrcnn_nuscenes_init.pth
"""
import argparse
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# NuScenes 模型参数（与原始 ONNX 一致）
NUM_CLASSES   = 10   # background(0) + 9 NuScenes foreground classes
INPUT_H       = 375
INPUT_W       = 1242

NUSCENES_CLASSES = [
    "background",             # 0
    "car",                    # 1
    "truck",                  # 2
    "bus",                    # 3
    "trailer",                # 4
    "construction_vehicle",   # 5
    "pedestrian",             # 6
    "motorcycle",             # 7
    "bicycle",                # 8
    "traffic_cone",           # 9
]

# torch hub 默认缓存路径
DEFAULT_COCO_CKPT = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
)


def build_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    构建 FasterRCNN ResNet50 FPN，检测头为 NuScenes 类别数。
    backbone 使用 COCO 预训练权重（在 build_model_from_coco 中加载），
    这里仅创建架构。
    """
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=False,
        image_mean=[0.0, 0.0, 0.0],   # /255 归一化，与原始 ONNX 导出一致
        image_std= [1.0, 1.0, 1.0],
    )
    # 替换分类头为 NuScenes 类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # 1024
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_model_from_coco(coco_ckpt: str, num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    ① 加载 COCO 预训练 FasterRCNN（91类）
    ② 替换检测头为 NuScenes num_classes 版本
    ③ backbone + FPN + RPN 权重完整保留
    """
    # 直接从 torch hub 加载 COCO 预训练模型
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=False,
        image_mean=[0.0, 0.0, 0.0],
        image_std= [1.0, 1.0, 1.0],
    )

    # 加载 COCO checkpoint（91 类）
    print(f"  加载 COCO 预训练权重: {coco_ckpt}")
    coco_sd = torch.load(coco_ckpt, map_location="cpu")
    # torchvision 的 COCO checkpoint 直接就是 state_dict
    model.load_state_dict(coco_sd, strict=False)

    # 替换分类器头（随机初始化 NuScenes 类别数）
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # 1024
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"  检测头替换完成: COCO 91类 → NuScenes {num_classes}类")

    return model


def verify_model(model: torch.nn.Module) -> None:
    """用随机输入验证模型可以推理。"""
    model.eval()
    dummy = torch.zeros(3, INPUT_H, INPUT_W)
    with torch.no_grad():
        out = model([dummy])
    print(f"  推理验证通过 ✅  boxes={out[0]['boxes'].shape} "
          f"labels={out[0]['labels'].shape} scores={out[0]['scores'].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_ckpt", default=DEFAULT_COCO_CKPT,
                        help="COCO 预训练 .pth 路径")
    parser.add_argument("--output",    default="fasterrcnn_nuscenes_init.pth")
    args = parser.parse_args()

    if not os.path.exists(args.coco_ckpt):
        print(f"[ERROR] COCO checkpoint 未找到: {args.coco_ckpt}")
        print("  请先运行以下命令下载：")
        print("  python3 -c \"import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)\"")
        return

    print(f"[1/3] 加载 COCO 预训练模型并替换检测头 (num_classes={NUM_CLASSES})...")
    model = build_model_from_coco(args.coco_ckpt, NUM_CLASSES)

    total_params  = sum(p.numel() for p in model.parameters())
    train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}  可训练: {train_params:,}")

    print("\n[2/3] 验证推理...")
    verify_model(model)

    print(f"\n[3/3] 保存初始化权重: {args.output}")
    torch.save({
        "state_dict":  model.state_dict(),
        "num_classes": NUM_CLASSES,
        "classes":     NUSCENES_CLASSES,
        "input_h":     INPUT_H,
        "input_w":     INPUT_W,
        "init_from":   "coco_pretrained",
    }, args.output)
    print(f"  已保存 ✅")
    print(f"\n下一步: python3 train.py --weights {args.output} --dataroot /data/nuscenes ...")


if __name__ == "__main__":
    main()

