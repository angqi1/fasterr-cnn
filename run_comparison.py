#!/usr/bin/env python3
"""
调用 C++ 节点运行4种配置，解析输出生成对比报告
配置组合: {FP16,INT8} × {threshold=0.5, threshold=0.25}
"""

import os, sys, subprocess, re, json, shutil, time
from pathlib import Path
from collections import defaultdict

WS          = Path("/home/nvidia/ros2_ws")
INSTALL     = WS / "install/faster_rcnn_ros"
NODE_BIN    = INSTALL / "lib/faster_rcnn_ros/faster_rcnn_node"
MODELS_DIR  = INSTALL / "share/faster_rcnn_ros/models"
LABELS      = MODELS_DIR / "labels.txt"
KITTI_DIR   = WS / "test_images/kitti_100/images"
OUTPUT_BASE = WS / "test_images/compare_results"

ROS2_SETUP  = WS / "install/setup.bash"

ENGINE_FP16 = MODELS_DIR / "faster_rcnn_500.engine"
ENGINE_INT8 = MODELS_DIR / "faster_rcnn_500_int8.engine"

CONFIGS = [
    ("FP16_thr0.5",  str(ENGINE_FP16), 0.5,  0.5),
    ("FP16_thr0.25", str(ENGINE_FP16), 0.25, 0.25),
    ("INT8_thr0.5",  str(ENGINE_INT8), 0.5,  0.5),
    ("INT8_thr0.25", str(ENGINE_INT8), 0.25, 0.25),
]

# ─────────────────────── 运行单个配置 ────────────────────────────────────────
def run_config(name, engine, threshold, ped_threshold):
    out_dir = OUTPUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash", "-c",
        f"source {ROS2_SETUP} && {NODE_BIN} --ros-args"
        f" -p engine_path:={engine}"
        f" -p labels_path:={LABELS}"
        f" -p input_path:={KITTI_DIR}"
        f" -p output_path:={out_dir}"
        f" -p input_height:=500"
        f" -p input_width:=1242"
        f" -p threshold:={threshold}"
        f" -p ped_threshold:={ped_threshold}"
    ]

    print(f"\n{'='*60}")
    print(f"  配置: {name} | engine={Path(engine).name} | thr={threshold}")
    print(f"  输出: {out_dir}")
    print(f"{'='*60}")

    t_start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600
    )
    t_total = time.time() - t_start

    output = result.stdout + result.stderr
    print(output[-3000:] if len(output) > 3000 else output)

    return parse_output(output, name, t_total)


# ─────────────────────── 解析节点输出 ────────────────────────────────────────
RESULT_RE = re.compile(r'\[(\S+\.png)\]\s+(\d+)\s+个目标\s+\|\s+([\d.]+)\s+ms')

def parse_output(text: str, cfg_name: str, total_sec: float) -> dict:
    matches = RESULT_RE.findall(text)
    if not matches:
        print(f"  [WARN] 未找到结果行，请检查节点输出")
        return {"cfg": cfg_name, "n_images": 0, "total_dets": 0, "avg_dets": 0,
                "lat_mean": 0, "lat_median": 0, "lat_p95": 0, "per_image": {}}

    import statistics
    latencies = [float(m[2]) for m in matches]
    dets_list = [int(m[1]) for m in matches]
    per_image = {m[0]: {"dets": int(m[1]), "ms": float(m[2])} for m in matches}

    return {
        "cfg"       : cfg_name,
        "n_images"  : len(matches),
        "total_dets": sum(dets_list),
        "avg_dets"  : sum(dets_list) / len(dets_list),
        "lat_mean"  : statistics.mean(latencies),
        "lat_median": statistics.median(latencies),
        "lat_p95"   : sorted(latencies)[int(len(latencies) * 0.95)],
        "per_image" : per_image,
    }


# ─────────────────────── 主流程 ──────────────────────────────────────────────
def main():
    assert NODE_BIN.exists(), f"节点不存在: {NODE_BIN}"
    assert ENGINE_FP16.exists(), f"FP16 引擎不存在: {ENGINE_FP16}"
    assert ENGINE_INT8.exists(), f"INT8 引擎不存在: {ENGINE_INT8}"

    images = sorted(KITTI_DIR.glob("*.png"))
    print(f"KITTI 图片: {len(images)} 张")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = []

    for name, engine, thr, ped_thr in CONFIGS:
        res = run_config(name, engine, thr, ped_thr)
        all_results.append(res)

    # ── 对比表 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ★ 实际推理对比结果（KITTI 100 张图片）")
    print(f"{'='*70}")

    SIZES = {
        "FP16": ENGINE_FP16.stat().st_size // 1024**2,
        "INT8": ENGINE_INT8.stat().st_size // 1024**2,
    }

    hdr = f"{'配置':<20} {'图片数':>6} {'总检出':>8} {'均检出/图':>10} {'延迟均值':>10} {'延迟P95':>9} {'引擎大小':>9}"
    print(hdr)
    print("-" * len(hdr))

    for r in all_results:
        prec = r["cfg"].split("_")[0]
        sz   = SIZES.get(prec, 0)
        print(f"{r['cfg']:<20} {r['n_images']:>6} {r['total_dets']:>8} "
              f"{r['avg_dets']:>10.1f} {r['lat_mean']:>9.1f}ms "
              f"{r['lat_p95']:>8.1f}ms {sz:>7}MB")

    # ── 详细类别分析（从结果图片目录推断）────────────────────────────────
    print(f"\n{'─'*70}")
    print("  延迟分布对比")
    print(f"{'─'*70}")
    for r in all_results:
        if r["n_images"] == 0:
            continue
        lats = sorted(r["per_image"][img]["ms"] for img in r["per_image"])
        p50  = lats[len(lats)//2]
        p99  = lats[int(len(lats)*0.99)] if len(lats) > 1 else lats[-1]
        print(f"  {r['cfg']:<20}: 最小={min(lats):.1f}ms  中位={p50:.1f}ms  "
              f"P95={r['lat_p95']:.1f}ms  最大={max(lats):.1f}ms")

    # ── FP16 vs INT8 对比分析 ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  FP16 vs INT8 精度影响（阈值相同，检出数差异）")
    print(f"{'─'*70}")
    res_map = {r["cfg"]: r for r in all_results}
    for thr_str in ["thr0.5", "thr0.25"]:
        fp16 = res_map.get(f"FP16_{thr_str}", {})
        int8 = res_map.get(f"INT8_{thr_str}", {})
        if fp16 and int8:
            diff = int8["total_dets"] - fp16["total_dets"]
            sign = "+" if diff >= 0 else ""
            lat_diff = int8["lat_mean"] - fp16["lat_mean"]
            lat_sign = "+" if lat_diff >= 0 else ""
            print(f"  {thr_str}: FP16={fp16['total_dets']} vs INT8={int8['total_dets']} "
                  f"(检出差 {sign}{diff})  "
                  f"延迟差 {lat_sign}{lat_diff:.1f}ms")

    # ── 保存结果 ─────────────────────────────────────────────────────────
    summary = {"configs": all_results, "engine_sizes_mb": SIZES}
    out_json = OUTPUT_BASE / "comparison_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n结果图片: {OUTPUT_BASE}/")
    print(f"汇总JSON: {out_json}")


if __name__ == "__main__":
    main()
