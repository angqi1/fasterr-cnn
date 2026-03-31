#!/usr/bin/env python3
"""
从 SSLAD-2D val 集均匀抽取 300 张图片，
将 COCO 格式标注转为逐图 KITTI 风格 txt（便于复用评估脚本）。

输出:
  test_images/sslad_300/images/    300 张图片（符号链接）
  test_images/sslad_300/labels/    300 个 txt 标注文件

txt 格式（与 KITTI 一致）:
  class trunc occl alpha x1 y1 x2 y2 dh dw dl dx dy dz ry
  其中只有 class, x1, y1, x2, y2 有实际值，其余填 0。

SSLAD → KITTI 类别映射:
  1:Pedestrian → Pedestrian
  2:Cyclist    → Cyclist
  3:Car        → Car
  4:Truck      → Truck
  5:Tram       → Tram
  6:Tricycle   → (忽略，NuScenes 模型无此类)
"""

import json, os, random
from pathlib import Path
from collections import defaultdict

random.seed(42)

WS = Path("/home/nvidia/ros2_ws")
VAL_DIR   = WS / "test_images/SSLAD-2D/labeled/val"
ANN_FILE  = WS / "test_images/SSLAD-2D/labeled/annotations/instance_val.json"
OUT_DIR   = WS / "test_images/sslad_300"
OUT_IMGS  = OUT_DIR / "images"
OUT_LBLS  = OUT_DIR / "labels"
N_SAMPLE  = 300

# SSLAD category_id → KITTI 类名
SSLAD_TO_KITTI = {
    1: "Pedestrian",
    2: "Cyclist",
    3: "Car",
    4: "Truck",
    5: "Tram",
    # 6: Tricycle → 忽略
}

def main():
    with open(ANN_FILE) as f:
        data = json.load(f)

    # 建立 image_id → image_info 映射
    id2img = {img["id"]: img for img in data["images"]}

    # 建立 image_id → [annotations] 映射
    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # 只选有标注的图片
    annotated_ids = sorted(img_anns.keys())
    print(f"有标注的图片: {len(annotated_ids)} / {len(data['images'])}")

    # 均匀抽取 300 张（等间隔采样，保证分布均匀）
    step = len(annotated_ids) / N_SAMPLE
    selected_ids = [annotated_ids[int(i * step)] for i in range(N_SAMPLE)]
    print(f"抽取: {len(selected_ids)} 张")

    # 统计抽样后的类别分布
    from collections import Counter
    cat_count = Counter()
    total_objs = 0
    for img_id in selected_ids:
        for ann in img_anns[img_id]:
            cid = ann["category_id"]
            if cid in SSLAD_TO_KITTI:
                cat_count[SSLAD_TO_KITTI[cid]] += 1
                total_objs += 1

    # 创建输出目录
    OUT_IMGS.mkdir(parents=True, exist_ok=True)
    OUT_LBLS.mkdir(parents=True, exist_ok=True)

    skipped_tricycle = 0
    for img_id in selected_ids:
        img_info = id2img[img_id]
        fname = img_info["file_name"]
        stem = Path(fname).stem

        # 创建图片符号链接
        src = VAL_DIR / fname
        dst = OUT_IMGS / fname
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)

        # 写 KITTI 风格 txt
        lines = []
        for ann in img_anns[img_id]:
            cid = ann["category_id"]
            if cid not in SSLAD_TO_KITTI:
                skipped_tricycle += 1
                continue
            cls_name = SSLAD_TO_KITTI[cid]
            # COCO bbox: [x, y, w, h] → KITTI: x1 y1 x2 y2
            bx, by, bw, bh = ann["bbox"]
            x1, y1 = bx, by
            x2, y2 = bx + bw, by + bh
            # KITTI 格式 15 列: class trunc occl alpha x1 y1 x2 y2 dh dw dl dx dy dz ry
            lines.append(f"{cls_name} 0 0 0 {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} 0 0 0 0 0 0 0")

        with open(OUT_LBLS / (stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"\n=== 完成 ===")
    print(f"图片: {OUT_IMGS}  ({len(selected_ids)} 张)")
    print(f"标注: {OUT_LBLS}  ({len(selected_ids)} 个 txt)")
    print(f"跳过 Tricycle 目标: {skipped_tricycle}")
    print(f"有效目标总数: {total_objs}")
    print(f"\n类别分布:")
    for cls, cnt in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"  {cls:>12}: {cnt}")

if __name__ == "__main__":
    main()
