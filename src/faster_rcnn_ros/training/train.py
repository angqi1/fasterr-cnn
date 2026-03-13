"""
train.py
NuScenes Faster RCNN 微调主脚本。

典型用法:
  # Step 1: 用 COCO 预训练权重初始化 NuScenes 检测头
  python3 extract_weights.py --output fasterrcnn_nuscenes_init.pth

  # Step 2: 下载 NuScenes mini（需在 nuscenes.org 注册后下载，约 4GB）
  #   https://www.nuscenes.org/download  → "Mini" → 解压到 /data/nuscenes

  # Step 3: 微调
  python3 train.py \
      --weights fasterrcnn_nuscenes_init.pth \
      --dataroot /data/nuscenes \
      --version v1.0-mini \
      --epochs 20 \
      --lr 0.001 \
      --output fasterrcnn_nuscenes_ft.pth

  # Step 4: 导出为 ONNX
  python3 export_onnx.py \
      --weights fasterrcnn_nuscenes_ft.pth \
      --output fasterrcnn_nuscenes_ft.onnx
"""
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader, random_split

from extract_weights import build_model, NUM_CLASSES, INPUT_H, INPUT_W
from nuscenes_dataset import NuScenesDataset, collate_fn
from transforms import get_train_transforms, get_val_transforms


def parse_args():
    p = argparse.ArgumentParser(description="NuScenes Faster RCNN 微调")
    p.add_argument("--weights",  required=True,  help="初始权重 .pth（来自 extract_weights.py）")
    p.add_argument("--dataroot", required=True,  help="NuScenes 数据集根目录")
    p.add_argument("--version",  default="v1.0-mini",  help="v1.0-mini / v1.0-trainval")
    p.add_argument("--cameras",  default="CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT",
                   help="逗号分隔的相机列表")
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--lr",       type=float, default=5e-4,  help="学习率")
    p.add_argument("--batch",    type=int,   default=2,     help="batch size")
    p.add_argument("--workers",  type=int,   default=2)
    p.add_argument("--val_split",type=float, default=0.15,  help="验证集比例")
    p.add_argument("--output",   default="fasterrcnn_nuscenes_ft.pth")
    p.add_argument("--save_every",type=int,  default=5,    help="每 N epoch 保存一次")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="冻结 backbone（只微调 FPN+Head）")
    return p.parse_args()


def get_lr_scheduler(optimizer, num_epochs: int):
    """余弦退火学习率调度。"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


def train_one_epoch(model, optimizer, dataloader, device, epoch: int):
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for i, (images, targets) in enumerate(dataloader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 过滤空 gt 的样本（避免 torchvision 报错）
        valid = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].numel() > 0]
        if not valid:
            continue
        images, targets = zip(*valid)

        loss_dict = model(list(images), list(targets))
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # 梯度裁剪，防止不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"  [epoch {epoch}] [{i+1}/{n_batches}] "
                  f"loss={losses.item():.4f} "
                  f"(cls={loss_dict.get('loss_classifier',0):.3f} "
                  f"box={loss_dict.get('loss_box_reg',0):.3f} "
                  f"rpn_cls={loss_dict.get('loss_objectness',0):.3f} "
                  f"rpn_box={loss_dict.get('loss_rpn_box_reg',0):.3f})")

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold: float = 0.3):
    """简单的检测数量统计作为快速验证指标（无完整 mAP）。"""
    model.eval()
    total_gt = 0
    total_pred = 0
    n = 0

    for images, targets in dataloader:
        images  = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            total_gt   += tgt["boxes"].shape[0]
            pred_scores = out["scores"]
            total_pred += (pred_scores >= threshold).sum().item()
        n += len(images)

    print(f"  [val] {n} 张图 | avg GT={total_gt/max(1,n):.1f} | "
          f"avg pred(>{threshold})={total_pred/max(1,n):.1f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 1. 加载模型 ────────────────────────────────────────────────────────
    print(f"\n[1] 加载权重: {args.weights}")
    ckpt = torch.load(args.weights, map_location="cpu")
    model = build_model(NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # 可选：冻结 backbone 只微调 FPN + Head（更快收敛，适合数据量小的情况）
    if args.freeze_backbone:
        print("  冻结 backbone.body 参数")
        for p in model.backbone.body.parameters():
            p.requires_grad_(False)

    # 可训练参数数量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {trainable:,}")

    # ── 2. 数据集 ──────────────────────────────────────────────────────────
    cameras = args.cameras.split(",")
    print(f"\n[2] 加载数据集: {args.dataroot} ({args.version})")
    print(f"    相机: {cameras}")

    full_dataset = NuScenesDataset(
        dataroot=args.dataroot,
        version=args.version,
        cameras=cameras,
        transforms=None,   # 先不加增强，split 后再加
    )

    n_val  = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # 为训练集添加增强（通过 wrapper 注入 transforms）
    class WithTransform(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            img, tgt = self.base[i]
            return self.transform(img, tgt)

    train_ds = WithTransform(train_ds, get_train_transforms(INPUT_H, INPUT_W))
    val_ds   = WithTransform(val_ds,   get_val_transforms(INPUT_H, INPUT_W))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn)

    print(f"  训练集: {n_train} | 验证集: {n_val}")

    # ── 3. 优化器 ──────────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, args.epochs)

    # ── 4. 训练循环 ────────────────────────────────────────────────────────
    best_loss = float("inf")
    print(f"\n[3] 开始训练 ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"\nepoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
        evaluate(model, val_loader, device)

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "state_dict":  model.state_dict(),
                "num_classes": NUM_CLASSES,
                "epoch":       epoch,
                "loss":        avg_loss,
            }, args.output.replace(".pth", "_best.pth"))
            print(f"  → 保存最优模型 (loss={avg_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = args.output.replace(".pth", f"_epoch{epoch:03d}.pth")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"  → checkpoint: {ckpt_path}")

    # 最终保存
    torch.save({
        "state_dict":  model.state_dict(),
        "num_classes": NUM_CLASSES,
        "epochs":      args.epochs,
    }, args.output)
    print(f"\n训练完成！最终模型: {args.output}")
    print(f"最优模型: {args.output.replace('.pth', '_best.pth')} (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
