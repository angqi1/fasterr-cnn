#!/usr/bin/env python3
"""
download_nuscenes_mini.py — nuScenes mini 数据集下载辅助脚本

nuScenes 数据集需要免费注册后才能下载。本脚本提供两种方式：
  方式A（推荐）: 手动从官网下载，解压到指定目录
  方式B:         如已有下载链接（登录后获取的签名URL），直接在脚本中填入

用法:
  python3 download_nuscenes_mini.py --check          # 检查本地数据是否就绪
  python3 download_nuscenes_mini.py --instructions   # 显示下载步骤
  python3 download_nuscenes_mini.py --url <signed_url>  # 用签名URL直接下载
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEST_DIR = Path("/home/nvidia/ros2_ws/test_images/nuscenes")

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║         nuScenes mini 数据集下载步骤                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  为什么用 nuScenes 而不是 KITTI？                                     ║
║  ─────────────────────────────────────────────────────────────────  ║
║  • 模型(fasterrcnn_nuscenes.onnx)在 NuScenes 上训练                  ║
║  • NuScenes 图像分辨率 1600×900，KITTI 仅 1224×370                   ║
║  • KITTI cyclist → 模型bicycle类 存在严重域偏移 → Recall=0%          ║
║  • NuScenes 使用相同10类标注，无需类别映射转换                        ║
║                                                                      ║
║  下载步骤（约 4 GB，免费注册）：                                      ║
║  ─────────────────────────────────────────────────────────────────  ║
║  1. 打开浏览器访问: https://www.nuscenes.org/nuscenes               ║
║  2. 点击右上角 "Free Download" → 注册/登录账户                       ║
║  3. 勾选同意使用协议，在 Download 页面找到:                          ║
║       "v1.0-mini" (~4.0 GB)                                          ║
║  4. 复制下载链接（或直接下载 .tgz 文件）                             ║
║  5. 将 .tgz 文件放到本机后执行解压命令:                              ║
║                                                                      ║
║     # 创建目录                                                        ║
║     mkdir -p ~/ros2_ws/test_images/nuscenes                          ║
║                                                                      ║
║     # 解压（替换为实际文件名/路径）                                   ║
║     tar -xzf v1.0-mini.tgz -C ~/ros2_ws/test_images/nuscenes/       ║
║                                                                      ║
║  6. 解压后目录结构应为:                                               ║
║     test_images/nuscenes/                                            ║
║       v1.0-mini/                                                     ║
║         scene.json                                                   ║
║         sample.json                                                  ║
║         sample_data.json                                             ║
║         sample_annotation.json                                       ║
║         category.json                                                ║
║         ...                                                          ║
║       samples/                                                       ║
║         CAM_FRONT/      ← 此处为评估用的图像                         ║
║         CAM_FRONT_LEFT/                                              ║
║         CAM_FRONT_RIGHT/                                             ║
║         ...                                                          ║
║                                                                      ║
║  7. 解压完成后运行检查:                                               ║
║     python3 download_nuscenes_mini.py --check                        ║
║                                                                      ║
║  8. 数据就绪后运行评估:                                               ║
║     python3 eval_nuscenes.py                                         ║
║                                                                      ║
║  mini 数据集规模:                                                     ║
║  • 10 个场景 × ~40 关键帧 = ~400 个样本                              ║
║  • 每样本 6 个摄像头 → 共 ~2400 张图像                               ║
║  • 拼合 CAM_FRONT + CAM_FRONT_LEFT + CAM_FRONT_RIGHT → 约 1200 张  ║
║  • eval_nuscenes.py 默认取前 500 张用于评估                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def check_data():
    """检查本地 NuScenes 数据是否就绪，返回 (ok, dataroot) 或 (False, None)"""
    for cand in [DEST_DIR, Path("/data/nuscenes"), Path("/home/nvidia/nuscenes")]:
        meta = cand / "v1.0-mini" / "scene.json"
        imgs = cand / "samples" / "CAM_FRONT"
        if meta.exists() and imgs.exists():
            n_imgs = len(list(imgs.glob("*.jpg"))) + len(list(imgs.glob("*.png")))
            print(f"[✓] 找到 nuScenes mini 数据: {cand}")
            print(f"    元数据: {meta}")
            print(f"    CAM_FRONT 图像数: {n_imgs}")
            return True, str(cand)
    print("[✗] 未找到本地 nuScenes mini 数据")
    print(f"    期望路径: {DEST_DIR}/v1.0-mini/scene.json")
    return False, None


def download_from_url(signed_url: str):
    """使用官网签名URL直接下载"""
    tgz_path = DEST_DIR.parent / "v1.0-mini.tgz"
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[下载] {signed_url}")
    print(f"[目标] {tgz_path}  (约 4 GB，请耐心等待...)")
    ret = subprocess.call(["wget", "-O", str(tgz_path), "--show-progress", signed_url])
    if ret != 0:
        print("[ERROR] 下载失败，请检查URL是否有效（签名URL通常有时效限制）")
        sys.exit(1)

    print(f"\n[解压] 到 {DEST_DIR} ...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    ret = subprocess.call(["tar", "-xzf", str(tgz_path), "-C", str(DEST_DIR)])
    if ret != 0:
        print("[ERROR] 解压失败")
        sys.exit(1)

    ok, _ = check_data()
    if ok:
        print("\n[✓] nuScenes mini 数据准备完成！运行: python3 eval_nuscenes.py")
    else:
        print("\n[WARN] 解压完成但目录结构不符合预期，请手动检查")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check",        action="store_true", help="检查本地数据是否就绪")
    ap.add_argument("--instructions", action="store_true", help="显示下载步骤")
    ap.add_argument("--url",          default="",          help="官网签名下载URL")
    args = ap.parse_args()

    if args.check:
        ok, dr = check_data()
        sys.exit(0 if ok else 1)
    elif args.instructions or not args.url:
        print(INSTRUCTIONS)
        ok, dr = check_data()
        if ok:
            print("\n  → 数据已就绪，直接运行: python3 eval_nuscenes.py")
    elif args.url:
        download_from_url(args.url)


if __name__ == "__main__":
    main()
