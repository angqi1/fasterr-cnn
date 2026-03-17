#!/usr/bin/env python3
"""
从官方 KITTI Object Detection 数据集中按需下载子集（图片+标签）。

特性：
- 使用远程 ZIP 按需解压，不下载整包
- 默认下载 100 张（可调）
- 输出为 KITTI 常见目录结构：
  <out_root>/images/000000.png
  <out_root>/labels/000000.txt

示例:
  /usr/bin/python download_kitti_subset.py --count 120 --out-root test_images/kitti_120
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

from remotezip import RemoteZip

IMG_ZIP_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
LBL_ZIP_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"


def list_members(url: str, prefix: str, suffix: str) -> Dict[str, str]:
    """返回 stem -> zip_member_path 映射。"""
    mapping: Dict[str, str] = {}
    with RemoteZip(url) as zf:
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(suffix):
                stem = Path(name).stem
                mapping[stem] = name
    return mapping


def extract_selected(url: str, members: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with RemoteZip(url) as zf:
        for idx, member in enumerate(members, 1):
            data = zf.read(member)
            out_path = out_dir / Path(member).name
            with open(out_path, "wb") as f:
                f.write(data)
            if idx % 20 == 0 or idx == len(members):
                print(f"  已下载 {idx}/{len(members)} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="下载样本数量，默认100")
    parser.add_argument("--out-root", default="test_images/kitti_100", help="输出根目录")
    parser.add_argument("--start-index", type=int, default=0, help="从排序后的第几个样本开始")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count 必须 > 0")

    out_root = Path(args.out_root)
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"

    print("扫描远程图片索引...")
    img_map = list_members(IMG_ZIP_URL, "training/image_2/", ".png")
    print("扫描远程标签索引...")
    lbl_map = list_members(LBL_ZIP_URL, "training/label_2/", ".txt")

    stems = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    if args.start_index >= len(stems):
        raise ValueError(f"--start-index 超出范围，最多 {len(stems)-1}")

    selected_stems = stems[args.start_index: args.start_index + args.count]
    if len(selected_stems) < args.count:
        print(f"警告: 可用样本仅 {len(selected_stems)}，小于请求的 {args.count}")

    img_members = [img_map[s] for s in selected_stems]
    lbl_members = [lbl_map[s] for s in selected_stems]

    print(f"开始下载 {len(selected_stems)} 个样本到: {out_root}")
    extract_selected(IMG_ZIP_URL, img_members, img_dir)
    extract_selected(LBL_ZIP_URL, lbl_members, lbl_dir)

    n_img = len(list(img_dir.glob("*.png")))
    n_lbl = len(list(lbl_dir.glob("*.txt")))
    print("下载完成")
    print(f"  图片: {n_img}")
    print(f"  标签: {n_lbl}")
    print(f"  图片目录: {img_dir}")
    print(f"  标签目录: {lbl_dir}")


if __name__ == "__main__":
    main()
