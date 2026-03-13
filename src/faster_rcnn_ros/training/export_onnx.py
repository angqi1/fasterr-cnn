"""
export_onnx.py
将微调后的 .pth 权重导出为 ONNX，供 build_engine.py 继续处理。

用法:
  python3 export_onnx.py \
      --weights fasterrcnn_nuscenes_ft.pth \
      --output  fasterrcnn_nuscenes_ft.onnx
"""
import argparse
import torch
import onnx
from extract_weights import build_model, NUM_CLASSES, INPUT_H, INPUT_W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="微调后的 .pth 文件")
    parser.add_argument("--output",  default="fasterrcnn_nuscenes_ft.onnx")
    parser.add_argument("--opset",   type=int, default=13)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] 加载权重: {args.weights}  (device={device})")
    ckpt  = torch.load(args.weights, map_location="cpu")
    model = build_model(NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    print(f"[2] 导出 ONNX (opset={args.opset}, input={INPUT_H}×{INPUT_W})...")
    dummy = torch.zeros(3, INPUT_H, INPUT_W, device=device)

    # torchvision Faster RCNN 需要输入为 List[Tensor]
    with torch.no_grad():
        torch.onnx.export(
            model,
            ([dummy],),
            args.output,
            opset_version=args.opset,
            input_names=["image"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={
                "image":  {1: "height", 2: "width"},
                "boxes":  {0: "num_detections"},
                "labels": {0: "num_detections"},
                "scores": {0: "num_detections"},
            },
            do_constant_folding=True,
        )

    # 验证 ONNX 有效性
    model_onnx = onnx.load(args.output)
    onnx.checker.check_model(model_onnx)
    print(f"  ONNX 验证通过 ✅")

    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"  输出文件: {args.output} ({size_mb:.1f} MB)")
    print(f"\n[3] 下一步: 运行 build_engine.py 将 ONNX 转为 TRT Engine")
    print(f"  cd ../models && python3 build_engine.py --onnx ../training/{args.output}")


if __name__ == "__main__":
    main()
