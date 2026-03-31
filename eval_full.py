#!/usr/bin/env python3
"""
eval_full.py  — 全量对比评估（含逐类 Recall/Precision 明细）

用法:
  python3 eval_full.py                     # 使用 kitti_500（若不足则用 kitti_100）
  python3 eval_full.py --data-dir test_images/kitti_100
  python3 eval_full.py --max-images 100    # 限制图片数量

测试引擎:
  1. FP16_375h  thr=0.3        （基准-训练原图尺寸最小引擎）
  2. FP16_500h  thr=0.3        （基准-中等尺寸）
  3. INT8_500h  thr=0.3        （基准-INT8 快速）
  4. FP16_700h  thr=0.3        （新-训练分辨率匹配）
  5. FP16_700h  thr=0.15       （新-低阈值探测）
  6. INT8_500h  veh=0.3/ped=0.15  （按类别阈值优化）

逐类指标: car, truck, bus, pedestrian, bicycle（对应 KITTI 标注类别）
"""

import argparse, os, sys, time
from pathlib import Path
import numpy as np
import cv2
import ctypes

_cudart = ctypes.cdll.LoadLibrary("libcudart.so")
def _malloc(n):
    p = ctypes.c_void_p()
    assert _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    return p.value
def _free(p): _cudart.cudaFree(ctypes.c_void_p(p))
def _h2d(d, a): _cudart.cudaMemcpy(ctypes.c_void_p(d), a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes), ctypes.c_int(1))
def _d2h(a, s): _cudart.cudaMemcpy(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(s), ctypes.c_size_t(a.nbytes), ctypes.c_int(2))

import tensorrt as trt
MAX_DET = 2000

# ─── 路径 ─────────────────────────────────────────────────────────────────────
WS         = Path("/home/nvidia/ros2_ws")
MODELS_DIR = WS / "install/faster_rcnn_ros/share/faster_rcnn_ros/models"

# ─── 类别定义 ─────────────────────────────────────────────────────────────────
# NuScenes label_id → 名称
LABEL_NAMES = {
    1: "car", 2: "truck", 3: "bus", 4: "trailer",
    5: "constr_veh", 6: "pedestrian", 7: "motorcycle", 8: "bicycle",
    9: "traffic_cone", 10: "barrier",
}

# KITTI 类别 → NuScenes label_id（None=忽略）
KITTI_MAP = {
    "car": 1, "van": 1, "truck": 2, "bus": 3, "tram": 3,
    "pedestrian": 6, "person_sitting": 6,
    "cyclist": 8,
    "misc": None, "dontcare": None,
}

# GT 类别兼容表（用于命中匹配，允许跨近似类别）
COMPAT = {1:{1}, 2:{1,2}, 3:{2,3}, 6:{6}, 7:{7,8}, 8:{7,8}}

# KITTI 里实际出现 + 我们关注的类别
EVAL_CLASSES = {1: "car", 2: "truck", 3: "bus", 6: "pedestrian", 8: "bicycle"}
IOU_THR = 0.5

# ─── TRT 推理器 ────────────────────────────────────────────────────────────────
class Inferencer:
    def __init__(self, engine_path, input_h=500, input_w=1242):
        self.H, self.W = input_h, input_w
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path, "rb").read())
        self._ctx = self._eng.create_execution_context()
        self._ctx.set_input_shape("image", (1, 3, input_h, input_w))
        self._d_inp    = _malloc(3 * input_h * input_w * 4)
        self._d_scores = _malloc(MAX_DET * 4)
        self._d_labels = _malloc(MAX_DET * 4)
        self._d_boxes  = _malloc(MAX_DET * 4 * 4)
        self._ctx.set_tensor_address("image",  self._d_inp)
        self._ctx.set_tensor_address("scores", self._d_scores)
        self._ctx.set_tensor_address("labels", self._d_labels)
        self._ctx.set_tensor_address("boxes",  self._d_boxes)

    def infer(self, bgr):
        blob = cv2.dnn.blobFromImage(bgr, 1.0/255.0, (self.W, self.H), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        _h2d(self._d_inp, blob)
        self._ctx.execute_async_v3(0)
        _cudart.cudaDeviceSynchronize()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n <= 0:
            return np.empty(0, np.float32), np.empty(0, np.int32), np.empty((0, 4), np.float32)
        sc = np.empty(n, np.float32);  _d2h(sc, self._d_scores)
        lb = np.empty(n, np.int32);    _d2h(lb, self._d_labels)
        bx = np.empty(n * 4, np.float32); _d2h(bx, self._d_boxes)
        return sc, lb, bx.reshape(n, 4)

    def close(self):
        _free(self._d_inp); _free(self._d_scores); _free(self._d_labels); _free(self._d_boxes)


# ─── GT ──────────────────────────────────────────────────────────────────────
def load_gt(lbl_dir, stem):
    """返回 [(cls_id, x1,y1,x2,y2), ...]"""
    lbl = Path(lbl_dir) / (stem + ".txt")
    result = []
    if not lbl.exists():
        return result
    for line in lbl.read_text().splitlines():
        p = line.strip().split()
        if len(p) < 8:
            continue
        cls_id = KITTI_MAP.get(p[0].lower())
        if cls_id is None:
            continue
        result.append((cls_id, float(p[4]), float(p[5]), float(p[6]), float(p[7])))
    return result


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_gt(gt_list, pred_list):
    """按 score 降序贪心匹配，返回 (matched_gt_indices, matched_pred_indices)"""
    matched_gt = set(); matched_pred = set()
    for pi, pred in sorted(enumerate(pred_list), key=lambda x: -x[1][5]):
        pcls_compat = COMPAT.get(pred[0], {pred[0]})
        pb = pred[1:5]
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_list):
            if gi in matched_gt:
                continue
            gcls = gt[0]
            if gcls not in pcls_compat and pred[0] not in COMPAT.get(gcls, {gcls}):
                continue
            v = iou(pb, gt[1:5])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= IOU_THR and best_gi >= 0:
            matched_gt.add(best_gi); matched_pred.add(pi)
    return matched_gt, matched_pred


# ─── 逐图推理 + 收集 ──────────────────────────────────────────────────────────
def run_config(inf: Inferencer, images, lbl_dir, thr_fn):
    """
    thr_fn(label_id, score) → bool
    返回:
      per_image: {stem: [(cls_id, x1,y1,x2,y2, score), ...]}
      latencies: [ms, ...]
    """
    dummy = cv2.imread(str(images[0]))
    for _ in range(3):
        inf.infer(dummy)

    per_image = {}
    latencies = []
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        oh, ow = bgr.shape[:2]
        sx, sy = ow / inf.W, oh / inf.H
        t0 = time.perf_counter()
        scores, labels, boxes = inf.infer(bgr)
        latencies.append((time.perf_counter() - t0) * 1000)
        dets = []
        for i in range(len(scores)):
            sc = float(scores[i]); lb = int(labels[i])
            if not thr_fn(lb, sc):
                continue
            x1 = float(boxes[i, 0] * sx); y1 = float(boxes[i, 1] * sy)
            x2 = float(boxes[i, 2] * sx); y2 = float(boxes[i, 3] * sy)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append((lb, x1, y1, x2, y2, sc))
        per_image[img_path.stem] = dets
    return per_image, latencies


# ─── 汇总指标（总体 + 逐类）────────────────────────────────────────────────────
def compute_metrics(images, all_gt, per_image):
    """
    返回:
      overall: (gt, pred, hit, miss, fp, rec%, prec%)
      per_cls: {cls_id: (gt, pred, hit, miss, fp, rec%, prec%)}
    """
    # 总体
    tot_gt = tot_pred = tot_hit = tot_miss = tot_fp = 0
    # 逐类
    cls_gt   = {c: 0 for c in EVAL_CLASSES}
    cls_pred = {c: 0 for c in EVAL_CLASSES}
    cls_hit  = {c: 0 for c in EVAL_CLASSES}

    for img in images:
        stem = img.stem
        gt_list   = all_gt.get(stem, [])
        pred_list = per_image.get(stem, [])
        matched_gt_idx, matched_pred_idx = match_gt(gt_list, pred_list)

        hit  = len(matched_gt_idx)
        miss = len(gt_list) - hit
        fp   = len(pred_list) - len(matched_pred_idx)
        tot_gt   += len(gt_list)
        tot_pred += len(pred_list)
        tot_hit  += hit
        tot_miss += miss
        tot_fp   += fp

        # 逐类 GT 命中统计
        for gi, gt in enumerate(gt_list):
            c = gt[0]
            if c in cls_gt:
                cls_gt[c] += 1
                if gi in matched_gt_idx:
                    cls_hit[c] += 1

        # 逐类 Pred 统计（用于 Precision）
        for pi, pred in enumerate(pred_list):
            c = pred[0]
            # pred 类别映射到 EVAL_CLASSES（compat）
            for ec in EVAL_CLASSES:
                if c in COMPAT.get(ec, {ec}) or ec in COMPAT.get(c, {c}):
                    cls_pred[ec] += 1
                    break

    def safe(h, d): return h / d * 100 if d > 0 else 0.0
    overall = (tot_gt, tot_pred, tot_hit, tot_miss, tot_fp,
               safe(tot_hit, tot_gt), safe(tot_hit, tot_pred))

    per_cls = {}
    for c in EVAL_CLASSES:
        g = cls_gt[c]; ph = cls_hit[c]; p = cls_pred[c]
        m = g - ph; fp_c = max(0, p - ph)
        per_cls[c] = (g, p, ph, m, fp_c, safe(ph, g), safe(ph, p))

    return overall, per_cls


# ─── 打印 ─────────────────────────────────────────────────────────────────────
def print_config_result(cfg_name, overall, per_cls, latencies):
    gt, pred, hit, miss, fp, rec, prec = overall
    lat_mean = float(np.mean(latencies))
    lat_p95  = float(np.percentile(latencies, 95))
    print(f"\n  {'─'*65}")
    print(f"  ▶ {cfg_name}")
    print(f"    整体: GT={gt}  Pred={pred}  Hit={hit}  Miss={miss}  FP={fp}")
    print(f"    整体: Recall={rec:.1f}%  Precision={prec:.1f}%  "
          f"F1={2*rec*prec/(rec+prec):.1f}%  |  延迟 mean={lat_mean:.1f}ms  P95={lat_p95:.1f}ms")
    print(f"    {'类别':12s} {'GT':>5} {'Pred':>6} {'Hit':>5} {'Miss':>5} {'FP':>5} "
          f"{'Recall':>8} {'Prec':>8}")
    print(f"    {'─'*62}")
    for c, cname in EVAL_CLASSES.items():
        cg, cp, ch, cm, cfp, cr, cpr = per_cls[c]
        bar = "▇" * int(cr / 5)  # 每格5%
        print(f"    {cname:12s} {cg:5d} {cp:6d} {ch:5d} {cm:5d} {cfp:5d} "
              f"{cr:7.1f}%  {cpr:7.1f}%  {bar}")
    return lat_mean, lat_p95, rec, prec


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default="", help="KITTI 数据目录（含 images/ labels/）")
    ap.add_argument("--max-images", type=int, default=500, help="最多使用多少张图片")
    args = ap.parse_args()

    # 自动选择数据目录
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif (WS / "test_images/kitti_500/images").exists():
        n500 = len(list((WS / "test_images/kitti_500/images").glob("*.png")))
        if n500 >= 200:
            data_dir = WS / "test_images/kitti_500"
            print(f"[数据集] 使用 kitti_500（{n500} 张）")
        else:
            data_dir = WS / "test_images/kitti_100"
            print(f"[数据集] kitti_500 仅 {n500} 张，暂用 kitti_100（下载中...）")
    else:
        data_dir = WS / "test_images/kitti_100"
        print("[数据集] 使用 kitti_100")

    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    images = sorted(img_dir.glob("*.png"))[:args.max_images]
    print(f"图片数: {len(images)}  GT目录: {lbl_dir}")

    all_gt = {img.stem: load_gt(lbl_dir, img.stem) for img in images}
    total_gt = sum(len(v) for v in all_gt.values())
    print(f"GT 总目标: {total_gt}  (IoU阈值: {IOU_THR})\n")

    # ── 配置表 ──────────────────────────────────────────────────────────────
    PED_CLASSES = {6, 7, 8}
    CONFIGS = [
        # (名称,  引擎文件名,   高度,  threshold_fn)
        ("FP16_375h  thr=0.3  [基准]",
            "faster_rcnn_375.engine",  375, lambda lb, sc: sc >= 0.30),
        ("FP16_500h  thr=0.3  [基准]",
            "faster_rcnn_500.engine",  500, lambda lb, sc: sc >= 0.30),
        ("INT8_500h  thr=0.3  [基准]",
            "faster_rcnn_500_int8.engine", 500, lambda lb, sc: sc >= 0.30),
        ("FP16_700h  thr=0.3  [训练分辨率]",
            "faster_rcnn_700.engine",  700, lambda lb, sc: sc >= 0.30),
        ("FP16_700h  thr=0.15 [低阈值]",
            "faster_rcnn_700.engine",  700, lambda lb, sc: sc >= 0.15),
        ("INT8_500h  veh=0.3/ped=0.15 [按类别阈值]",
            "faster_rcnn_500_int8.engine", 500,
            lambda lb, sc, p=PED_CLASSES: sc >= (0.15 if lb in p else 0.30)),
    ]

    summary = []  # [(cfg_name, rec, prec, f1, lat_mean, lat_p95, engine_mb, per_cls)]

    for cfg_name, engine_file, inp_h, thr_fn in CONFIGS:
        engine_path = MODELS_DIR / engine_file
        if not engine_path.exists():
            print(f"[SKIP] {cfg_name}: {engine_file} 不存在")
            continue
        engine_mb = engine_path.stat().st_size // 1024 ** 2
        inf = Inferencer(str(engine_path), inp_h, 1242)
        per_image, latencies = run_config(inf, images, lbl_dir, thr_fn)
        inf.close()

        overall, per_cls = compute_metrics(images, all_gt, per_image)
        lat_mean, lat_p95, rec, prec = print_config_result(cfg_name, overall, per_cls, latencies)
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        summary.append((cfg_name, rec, prec, f1, lat_mean, lat_p95, engine_mb, per_cls, overall))

    # ── 整体对比表 ────────────────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"  ★ 综合对比表  ({len(images)} 张图 | IoU≥{IOU_THR})")
    print(f"{'═'*100}")
    hdr = f"  {'配置':40s} {'MB':>4} {'GT':>5} {'Pred':>6} {'Hit':>5} {'FP':>6} {'Recall':>8} {'Prec':>7} {'F1':>7} {'延迟':>8} {'P95':>7}"
    print(hdr)
    print("  " + "-" * 98)
    for name, rec, prec, f1, lat, p95, mb, _, overall in summary:
        gt, pred, hit, miss, fp, _, _ = overall
        print(f"  {name:40s} {mb:4d} {gt:5d} {pred:6d} {hit:5d} {fp:6d} {rec:7.1f}% {prec:6.1f}% {f1:6.1f}% {lat:8.1f} {p95:7.1f}")
    print("  " + "-" * 98)

    # ── 逐类汇总对比 ──────────────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"  ★ 逐类 Recall 对比表（单位: %）")
    print(f"{'═'*100}")
    cls_hdr = f"  {'配置':40s}" + "".join(f" {EVAL_CLASSES[c]:>12s}" for c in EVAL_CLASSES)
    print(cls_hdr)
    print("  " + "-" * 98)
    for name, rec, prec, f1, lat, p95, mb, per_cls, _ in summary:
        row = f"  {name:40s}"
        for c in EVAL_CLASSES:
            cr = per_cls[c][5]
            row += f" {cr:11.1f}%"
        print(row)
    print("  " + "-" * 98)

    print(f"\n  ★ 逐类 Precision 对比表（单位: %）")
    print("  " + "-" * 98)
    print(cls_hdr)
    print("  " + "-" * 98)
    for name, rec, prec, f1, lat, p95, mb, per_cls, _ in summary:
        row = f"  {name:40s}"
        for c in EVAL_CLASSES:
            cpr = per_cls[c][6]
            row += f" {cpr:11.1f}%"
        print(row)
    print("  " + "-" * 98)

    # ── GT 分布 ──────────────────────────────────────────────────────────────
    if summary:
        _, _, _, _, _, _, _, per_cls0, _ = summary[0]
        print(f"\n  GT 类别分布:")
        for c, cname in EVAL_CLASSES.items():
            g = per_cls0[c][0]
            pct = g / total_gt * 100
            print(f"    {cname:12s}: {g:4d} 个 ({pct:.1f}%)")

    # ── 最优方案推荐 ─────────────────────────────────────────────────────────
    if summary:
        best_rec  = max(summary, key=lambda x: x[1])
        best_f1   = max(summary, key=lambda x: x[3])
        best_spd  = min(summary, key=lambda x: x[4])
        print(f"\n  ★ 最高 Recall:  {best_rec[0]}  →  {best_rec[1]:.1f}%")
        print(f"  ★ 最高 F1:      {best_f1[0]}  →  F1={best_f1[3]:.1f}%")
        print(f"  ★ 最快推理:     {best_spd[0]}  →  {best_spd[4]:.1f}ms")

    import json
    out_json = WS / "test_images/compare_results/eval_full_summary.json"
    out_json.parent.mkdir(exist_ok=True)
    save_data = [
        {"config": n, "recall": r, "precision": p, "f1": f, "lat_mean": lm, "lat_p95": lp,
         "engine_mb": mb,
         "per_class_recall":     {EVAL_CLASSES[c]: pc[5] for c, pc in pcls.items()},
         "per_class_precision":  {EVAL_CLASSES[c]: pc[6] for c, pc in pcls.items()},
         "per_class_gt":         {EVAL_CLASSES[c]: pc[0] for c, pc in pcls.items()},
        }
        for n, r, p, f, lm, lp, mb, pcls, _ in summary
    ]
    out_json.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
    print(f"\n  结果已保存: {out_json}")


if __name__ == "__main__":
    main()
