#!/usr/bin/env python3
"""
download_nuimages_uniform.py — 从 nuImages CAM_FRONT sweeps tar 按时间均匀采样 500 张

工作原理:
  1. 下载 metadata TGZ (639 MB) → 解析 sample_data.json
  2. 过滤出所有 CAM_FRONT sweep 帧，按 timestamp 排序
  3. 将时间段均分 500 等份，每份取一帧 → 得到 500 个目标文件名
  4. 流式下载 sweeps TGZ (36.5 GB)，只写入目标文件，找完即停
  5. 保存图像和时间采样报告

目标: 覆盖数据集全部时间跨度（含新加坡/波士顿不同场景不同天）

用法:
  python3 download_nuimages_uniform.py --count 500
  python3 download_nuimages_uniform.py --count 500 --skip-meta  # 跳过已有targets.json

输出目录: ~/ros2_ws/test_images/nuimages_uniform/images/
"""

import argparse, datetime, json, os, sys, signal, tarfile, urllib.request
from pathlib import Path

URL_META   = ("https://motional-nuscenes.s3.amazonaws.com/public/"
              "nuimages-v1.0/nuimages-v1.0-all-metadata.tgz")
URL_SWEEPS = ("https://motional-nuscenes.s3.amazonaws.com/public/"
              "nuimages-v1.0/nuimages-v1.0-all-sweeps-cam-front.tgz")

WS          = Path("/home/nvidia/ros2_ws")
DEST_DIR    = WS / "test_images/nuimages_uniform/images"
TARGETS_JSON = WS / "test_images/nuimages_uniform/targets.json"
CHUNK       = 4 * 1024 * 1024   # 4 MB 缓冲


# ─── 流式读 wrapper ────────────────────────────────────────────────────────────
class StreamReader:
    """将 urllib Response 包装成 tarfile 可接受的流式 file-like 对象"""
    def __init__(self, resp):
        self._resp  = resp
        self._buf   = b""
        self._bytes = 0
        self._done  = False

    def read(self, n):
        while len(self._buf) < n and not self._done:
            chunk = self._resp.read(CHUNK)
            if not chunk:
                self._done = True
                break
            self._buf   += chunk
            self._bytes += len(chunk)
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def readable(self):  return True
    def seekable(self):  return False
    def writable(self):  return False
    @property
    def downloaded_mb(self): return self._bytes / 1024 / 1024


# ─── 第一步：解析 metadata，生成均匀采样目标 ────────────────────────────────────
def build_targets(count: int) -> dict:
    """
    返回:
      {basename: timestamp, ...}  (500 个目标，key 为纯文件名不含路径)
    """
    print(f"[步骤1] 下载 metadata ({URL_META.split('/')[-1]}, 约 639 MB)...")
    req  = urllib.request.Request(URL_META, headers={"User-Agent": "Mozilla/5.0"})
    data = None
    with urllib.request.urlopen(req) as resp:
        reader = StreamReader(resp)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for m in tar:
                if "sample_data" in m.name and m.name.endswith(".json"):
                    print(f"  解析 {m.name}  ({m.size/1024/1024:.0f} MB JSON) ...",
                          flush=True)
                    f    = tar.extractfile(m)
                    data = json.loads(f.read())
                    print(f"  总条目: {len(data)}", flush=True)
                    break

    if data is None:
        raise RuntimeError("未在 metadata 中找到 sample_data.json")

    # 过滤 CAM_FRONT sweep 帧（filename 含 /CAM_FRONT/，且 is_key_frame=False）
    cam = [d for d in data
           if "/CAM_FRONT/" in d.get("filename", "")
           and not d.get("is_key_frame", True)]
    print(f"  CAM_FRONT sweep 帧: {len(cam)}", flush=True)

    if not cam:
        raise RuntimeError("未找到 CAM_FRONT sweep 帧，请检查 metadata 字段")

    # 按时间戳排序
    cam.sort(key=lambda x: x["timestamp"])

    # 时间范围
    ts0 = cam[0]["timestamp"];  ts1 = cam[-1]["timestamp"]
    dt0 = datetime.datetime.fromtimestamp(ts0 / 1e6)
    dt1 = datetime.datetime.fromtimestamp(ts1 / 1e6)
    total_h = (ts1 - ts0) / 1e6 / 3600
    print(f"  时间范围: {dt0}  →  {dt1}  ({total_h:.1f}h)", flush=True)

    # 将时间段均分 count 份，每份取一帧（基于时间戳插值，而非固定步长）
    step  = (ts1 - ts0) / (count - 1)
    chosen = []
    idx   = 0
    for i in range(count):
        target_ts = ts0 + i * step
        # 找最近的帧
        while idx + 1 < len(cam) and abs(cam[idx + 1]["timestamp"] - target_ts) \
                                   < abs(cam[idx]["timestamp"]     - target_ts):
            idx += 1
        chosen.append(cam[idx])

    # 去重（相同文件名只保留一个，填补缺口）
    seen = {}
    for d in chosen:
        bn = Path(d["filename"]).name
        if bn not in seen:
            seen[bn] = d["timestamp"]

    # 如果去重后不足，用时间均匀补足
    if len(seen) < count:
        extra_needed = count - len(seen)
        seen_set     = set(seen.keys())
        step2        = len(cam) / (extra_needed * 2)
        for i in range(int(len(cam) / step2)):
            if len(seen) >= count:
                break
            d  = cam[int(i * step2)]
            bn = Path(d["filename"]).name
            if bn not in seen_set:
                seen[bn] = d["timestamp"]
                seen_set.add(bn)

    # 最终取前 count 个（按时间排序）
    result = dict(sorted(seen.items(), key=lambda x: x[1])[:count])
    print(f"  最终目标文件数: {len(result)}", flush=True)

    # 打印场景多样性统计
    scenes = set()
    for bn in result:
        parts = bn.split("__")
        if parts:
            scenes.add(parts[0])   # 例如 n003-2018-01-02-11-48-43+0800
    print(f"  覆盖不同 log (场景): {len(scenes)}", flush=True)
    print(f"  示例 log: {list(scenes)[:5]}", flush=True)

    return result


# ─── 第二步：流式提取目标图像 ──────────────────────────────────────────────────
def stream_extract_targets(targets: dict, dest: Path) -> int:
    """
    从 sweeps tar 流中只提取 targets 中的文件。
    targets: {basename: timestamp}
    返回实际提取数量。
    """
    dest.mkdir(parents=True, exist_ok=True)
    remaining = set(targets.keys())
    total     = len(remaining)
    extracted = 0

    # 跳过已存在的文件
    for bn in list(remaining):
        if (dest / bn).exists():
            remaining.discard(bn)
            extracted += 1

    if not remaining:
        print(f"[步骤2] 所有 {total} 张图像已存在，跳过下载", flush=True)
        return total

    print(f"[步骤2] 流式提取 sweeps tar (36.5 GB) ...", flush=True)
    print(f"  目标: {len(remaining)} 张 (已有: {extracted} 张)", flush=True)
    print(f"  输出: {dest}", flush=True)
    print(f"  提示: 需要流式扫描全部压缩包，耗时约 30-90 分钟（取决于网速）\n",
          flush=True)

    req = urllib.request.Request(URL_SWEEPS, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            reader = StreamReader(resp)
            with tarfile.open(fileobj=reader, mode="r|gz") as tar:
                for m in tar:
                    bn = Path(m.name).name
                    if bn not in remaining:
                        continue
                    f = tar.extractfile(m)
                    if f is None:
                        continue
                    (dest / bn).write_bytes(f.read())
                    remaining.discard(bn)
                    extracted += 1
                    pct = extracted / total * 100
                    mb  = reader.downloaded_mb
                    print(f"\r  [{extracted:4d}/{total}] {pct:5.1f}%  "
                          f"已下载: {mb:7.1f} MB  当前: {bn[:60]}",
                          end="", flush=True)
                    if not remaining:
                        print(f"\n  [完成] 所有目标文件已找到，提前结束流", flush=True)
                        break
    except (BrokenPipeError, KeyboardInterrupt):
        pass

    print(f"\n[步骤2] 提取完成: {extracted}/{total} 张  →  {dest}", flush=True)
    return extracted


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count",     type=int, default=500, help="目标图像数")
    ap.add_argument("--skip-meta", action="store_true",
                    help="跳过 metadata 下载，使用已有 targets.json")
    args = ap.parse_args()

    DEST_DIR.parent.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 步骤1: 生成目标列表
    if args.skip_meta and TARGETS_JSON.exists():
        print(f"[步骤1] 从缓存加载目标列表: {TARGETS_JSON}", flush=True)
        targets = json.loads(TARGETS_JSON.read_text())
    else:
        targets = build_targets(args.count)
        TARGETS_JSON.write_text(json.dumps(targets, indent=2))
        print(f"  目标列表已缓存: {TARGETS_JSON}", flush=True)

    # 步骤2: 流式提取
    n = stream_extract_targets(targets, DEST_DIR)

    # 完成报告
    imgs = list(DEST_DIR.glob("*.jpg")) + list(DEST_DIR.glob("*.png"))
    print(f"\n=== 完成 ===")
    print(f"图像目录:  {DEST_DIR}")
    print(f"实际图像数: {len(imgs)}")
    print(f"运行评估:   python3 eval_nuimages.py "
          f"--img-dir {DEST_DIR}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    main()
