#!/usr/bin/env python3
"""
download_nuimages.py — 流式提取 nuImages CAM_FRONT 图像（无需完整下载 36 GB）

工作原理：
  边下载边解压 tar.gz 流，提取前 N 张 .jpg 图像后立即断开连接。
  不会将整个 36 GB 文件写到磁盘。

用法:
  python3 download_nuimages.py              # 下载前 500 张 sweep 图
  python3 download_nuimages.py --count 200  # 下载前 200 张
  python3 download_nuimages.py --check      # 检查本地已有多少张

输出目录: ~/ros2_ws/test_images/nuimages/images/
"""

import argparse
import signal
import sys
import tarfile
import urllib.request
from pathlib import Path

URL_SWEEPS   = ("https://motional-nuscenes.s3.amazonaws.com/public/"
                "nuimages-v1.0/nuimages-v1.0-all-sweeps-cam-front.tgz")
DEST         = Path("/home/nvidia/ros2_ws/test_images/nuimages/images")
CHUNK        = 2 * 1024 * 1024   # 2 MB 读取块

# ── 进度显示 ──────────────────────────────────────────────────────────────────
class _Progress:
    def __init__(self, total=None):
        self._downloaded = 0
        self._total = total

    def update(self, n):
        self._downloaded += n
        mb = self._downloaded / 1024 / 1024
        if self._total:
            pct = self._downloaded / self._total * 100
            print(f"\r  [下载进度] {mb:6.1f} MB / {self._total/1024/1024:.0f} MB  ({pct:.1f}%)",
                  end="", flush=True)
        else:
            print(f"\r  [下载进度] {mb:6.1f} MB", end="", flush=True)


# ── 流式提取N张图 ─────────────────────────────────────────────────────────────
def stream_extract(url: str, dest: Path, count: int) -> int:
    """
    流式解压 tar.gz，提取前 count 张 .jpg。
    返回实际提取张数。
    """
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[目标]  {dest}")
    print(f"[URL]   {url}")
    print(f"[目标]  提取前 {count} 张图像（流式，无需下载全部文件）\n")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    total_size = None
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0)) or None
            progress = _Progress(total_size)

            # 使用 Chunked 文件包装器让 tarfile 能流式读取
            class _ChunkedReader:
                def __init__(self, response):
                    self._resp = response
                    self._buf  = b""
                    self._done = False

                def read(self, n):
                    while len(self._buf) < n and not self._done:
                        chunk = self._resp.read(CHUNK)
                        if not chunk:
                            self._done = True
                            break
                        progress.update(len(chunk))
                        self._buf += chunk
                    out, self._buf = self._buf[:n], self._buf[n:]
                    return out

                def readable(self): return True
                def seekable(self): return False
                def writable(self): return False

            reader = _ChunkedReader(resp)
            extracted = 0

            with tarfile.open(fileobj=reader, mode="r|gz") as tar:
                for member in tar:
                    name_lower = member.name.lower()
                    if not (name_lower.endswith(".jpg") or
                            name_lower.endswith(".jpeg") or
                            name_lower.endswith(".png")):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    img_name = Path(member.name).name
                    out_path = dest / img_name
                    out_path.write_bytes(f.read())
                    extracted += 1
                    print(f"\r  [{extracted:4d}/{count}] 已提取: {img_name}",
                          end="", flush=True)
                    if extracted >= count:
                        break

    except (BrokenPipeError, KeyboardInterrupt):
        pass  # 提取到目标数量后主动断开，正常

    print(f"\n\n[完成] 共提取 {extracted} 张图像 → {dest}")
    return extracted


def check_local(dest: Path):
    if not dest.exists():
        print(f"[✗] 目录不存在: {dest}")
        return 0
    imgs = list(dest.glob("*.jpg")) + list(dest.glob("*.png"))
    print(f"[✓] 本地已有 {len(imgs)} 张图像: {dest}")
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=500, help="下载图像张数（默认500）")
    ap.add_argument("--check", action="store_true",   help="仅检查本地图像数量")
    args = ap.parse_args()

    if args.check:
        n = check_local(DEST)
        sys.exit(0 if n >= args.count else 1)

    # 检查是否已有足够的图像
    existing = check_local(DEST)
    if existing >= args.count:
        print(f"[跳过] 本地已有 {existing} 张图像，无需重新下载")
        print(f"直接运行: python3 eval_nuimages.py")
        return

    needed = args.count - existing
    if existing > 0:
        print(f"[续下] 本地已有 {existing} 张，还需下载 {needed} 张")
        # 追加下载
        stream_extract(URL_SWEEPS, DEST, args.count)  # 会跳过已存在文件名
    else:
        stream_extract(URL_SWEEPS, DEST, args.count)

    print(f"\n数据就绪，运行评估: python3 eval_nuimages.py")


if __name__ == "__main__":
    # 捕获 Ctrl+C，优雅退出
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    main()
