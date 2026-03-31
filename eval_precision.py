#!/usr/bin/env python3
"""
eval_precision.py — 以 thr=0.5 为精度基线，对比 3 种优化方案

基线:  FP16_500h  thr=0.5                                         (精度基准)
方案1: 多尺度融合  FP16_375h + FP16_500h  thr=0.5  NMS合并        (↑召回，代价延迟)
方案2: INT8_500h  自适应阈值 bus=0.40/行人=0.35/自行车=0.25        (↑速度 + 难类别召回)
方案3: FP16_375h  thr=0.5                                         (↑速度优先)

用法: python3 eval_precision.py [--max-images 500]
"""

import argparse, time, json
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
def _h2d(d, a):
    _cudart.cudaMemcpy(ctypes.c_void_p(d), a.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_size_t(a.nbytes), ctypes.c_int(1))
def _d2h(a, s):
    _cudart.cudaMemcpy(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(s),
                       ctypes.c_size_t(a.nbytes), ctypes.c_int(2))

import tensorrt as trt

MAX_DET    = 2000
WS         = Path("/home/nvidia/ros2_ws")
MODELS_DIR = WS / "install/faster_rcnn_ros/share/faster_rcnn_ros/models"

# KITTI 类别 → NuScenes label_id
KITTI_MAP = {
    "car": 1, "van": 1, "truck": 2, "bus": 3, "tram": 3,
    "pedestrian": 6, "person_sitting": 6, "cyclist": 8,
    "misc": None, "dontcare": None,
}

# 评估时类别兼容表 (gt_cls → 允许匹配的 pred_cls 集合)
COMPAT = {1: {1}, 2: {1, 2}, 3: {2, 3}, 6: {6}, 7: {7, 8}, 8: {7, 8}}

EVAL_CLASSES = {1: "car", 2: "truck", 3: "bus", 6: "pedestrian", 8: "bicycle"}
IOU_THR      = 0.5

# ─── 方案2 自适应阈值表（NuScenes label_id → score 阈值）────────────────────
ADAPTIVE_THR = {
    1: 0.50,   # car
    2: 0.50,   # truck
    3: 0.40,   # bus       ← 难检，稍低
    4: 0.50,   # trailer
    5: 0.50,   # constr_veh
    6: 0.35,   # pedestrian ← 难检，降低
    7: 0.35,   # motorcycle
    8: 0.25,   # bicycle    ← 最难，进一步降低
    9: 0.50,   # traffic_cone
    10: 0.50,  # barrier
}

# ─── TRT 推理器 ────────────────────────────────────────────────────────────────
class Inferencer:
    def __init__(self, engine_path, input_h, input_w=1242):
        self.H, self.W = input_h, input_w
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(
            open(engine_path, "rb").read())
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
        blob = cv2.dnn.blobFromImage(
            bgr, 1.0 / 255.0, (self.W, self.H), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        _h2d(self._d_inp, blob)
        self._ctx.execute_async_v3(0)
        _cudart.cudaDeviceSynchronize()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n <= 0:
            return np.empty(0, np.float32), np.empty(0, np.int32), np.empty((0, 4), np.float32)
        sc = np.empty(n, np.float32);    _d2h(sc, self._d_scores)
        lb = np.empty(n, np.int32);      _d2h(lb, self._d_labels)
        bx = np.empty(n * 4, np.float32); _d2h(bx, self._d_boxes)
        return sc, lb, bx.reshape(n, 4)

    def close(self):
        _free(self._d_inp); _free(self._d_scores)
        _free(self._d_labels); _free(self._d_boxes)


# ─── GT 读取 ──────────────────────────────────────────────────────────────────
def load_gt(lbl_dir, stem):
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


# ─── IoU ─────────────────────────────────────────────────────────────────────
def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ─── 推理 + 坐标还原 + 过滤 ──────────────────────────────────────────────────
def do_infer(inf, bgr, thr_fn):
    """
    对单张图像做推理，将预测框映射回原图坐标，过滤低分预测。
    返回 [(cls, x1, y1, x2, y2, score), ...]
    """
    oh, ow = bgr.shape[:2]
    sx, sy  = ow / inf.W, oh / inf.H
    sc_arr, lb_arr, bx_arr = inf.infer(bgr)
    dets = []
    for i in range(len(sc_arr)):
        sc = float(sc_arr[i]); lb = int(lb_arr[i])
        if not thr_fn(lb, sc):
            continue
        x1 = float(bx_arr[i, 0] * sx); y1 = float(bx_arr[i, 1] * sy)
        x2 = float(bx_arr[i, 2] * sx); y2 = float(bx_arr[i, 3] * sy)
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append((lb, x1, y1, x2, y2, sc))
    return dets


# ─── 方案1：多尺度 NMS 合并 ────────────────────────────────────────────────────
def nms_merge(preds_list, iou_thr=0.5):
    """
    将多个引擎/尺度的预测列表合并，用 NMS 去除重复框（保留高分框）。
    同类别预测框 IoU > iou_thr 时视为同一目标，保留 score 更高的。
    """
    all_preds = [p for lst in preds_list for p in lst]
    all_preds.sort(key=lambda x: -x[5])
    used = [False] * len(all_preds)
    kept = []
    for i, p in enumerate(all_preds):
        if used[i]:
            continue
        used[i] = True
        kept.append(p)
        for j in range(i + 1, len(all_preds)):
            if used[j]:
                continue
            q = all_preds[j]
            compat = (p[0] == q[0]
                      or q[0] in COMPAT.get(p[0], {p[0]})
                      or p[0] in COMPAT.get(q[0], {q[0]}))
            if compat and iou(p[1:5], q[1:5]) >= iou_thr:
                used[j] = True
    return kept


# ─── GT 匹配 ──────────────────────────────────────────────────────────────────
def match_gt(gt_list, pred_list):
    """按 score 降序贪心匹配 GT，返回 (matched_gt_indices, matched_pred_indices)"""
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


# ─── 指标计算 ─────────────────────────────────────────────────────────────────
def compute_metrics(images, all_gt, per_image):
    tot_gt = tot_pred = tot_hit = 0
    cls_gt   = {c: 0 for c in EVAL_CLASSES}
    cls_pred = {c: 0 for c in EVAL_CLASSES}
    cls_hit  = {c: 0 for c in EVAL_CLASSES}

    for img in images:
        stem      = img.stem
        gt_list   = all_gt.get(stem, [])
        pred_list = per_image.get(stem, [])
        mgt, mpred = match_gt(gt_list, pred_list)

        tot_gt   += len(gt_list)
        tot_pred += len(pred_list)
        tot_hit  += len(mgt)

        for gi, gt in enumerate(gt_list):
            c = gt[0]
            if c in cls_gt:
                cls_gt[c] += 1
                if gi in mgt:
                    cls_hit[c] += 1

        for pred in pred_list:
            c = pred[0]
            for ec in EVAL_CLASSES:
                if c in COMPAT.get(ec, {ec}) or ec in COMPAT.get(c, {c}):
                    cls_pred[ec] += 1
                    break

    def safe(h, d): return h / d * 100 if d > 0 else 0.0
    tot_fp  = tot_pred - tot_hit
    overall = (tot_gt, tot_pred, tot_hit, tot_gt - tot_hit, tot_fp,
               safe(tot_hit, tot_gt), safe(tot_hit, tot_pred))
    per_cls = {}
    for c in EVAL_CLASSES:
        g = cls_gt[c]; ph = cls_hit[c]; p = cls_pred[c]
        per_cls[c] = (g, p, ph, g - ph, max(0, p - ph),
                      safe(ph, g), safe(ph, p))
    return overall, per_cls


# ─── 打印单条结果 ──────────────────────────────────────────────────────────────
def print_result(name, overall, per_cls, latencies):
    gt, pred, hit, miss, fp, rec, prec = overall
    lat_mean = float(np.mean(latencies))
    lat_p95  = float(np.percentile(latencies, 95))
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0
    SEP = "─" * 74
    print(f"\n  {SEP}")
    print(f"  ▶  {name}")
    print(f"     总体: GT={gt}  Pred={pred}  Hit={hit}  Miss={miss}  FP={fp}")
    print(f"     总体: Recall={rec:.1f}%  Precision={prec:.1f}%  F1={f1:.1f}%")
    print(f"     延迟: mean={lat_mean:.1f}ms  P95={lat_p95:.1f}ms")
    print(f"     {'类别':12s}{'GT':>5}{'Pred':>7}{'Hit':>6}{'Miss':>6}{'FP':>6}"
          f"{'Recall':>9}{'Prec':>9}")
    print(f"     {'─'*68}")
    for c, cname in EVAL_CLASSES.items():
        cg, cp, ch, cm, cfp, cr, cpr = per_cls[c]
        bar = "▇" * int(cr / 5)
        print(f"     {cname:12s}{cg:5d}{cp:7d}{ch:6d}{cm:6d}{cfp:6d}"
              f"{cr:8.1f}%{cpr:8.1f}%  {bar}")
    return lat_mean, lat_p95, rec, prec, f1


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=500,
                    help="最多使用图片数（默认 500）")
    args = ap.parse_args()

    # ── 选数据集（优先 kitti_500）──────────────────────────────────────────────
    data_dir = None
    for d_name in ["kitti_500", "kitti_100"]:
        cand = WS / f"test_images/{d_name}/images"
        if cand.exists():
            n = len(list(cand.glob("*.png")))
            if n >= 5:
                data_dir = WS / f"test_images/{d_name}"
                print(f"[数据集] 使用 {d_name}（{n} 张）")
                break
    if data_dir is None:
        print("[错误] 未找到测试数据集，请先运行 download_kitti_subset.py")
        return

    images  = sorted((data_dir / "images").glob("*.png"))[:args.max_images]
    lbl_dir = data_dir / "labels"
    print(f"图片数: {len(images)}  IoU评估阈值: {IOU_THR}")

    all_gt    = {img.stem: load_gt(lbl_dir, img.stem) for img in images}
    total_gt  = sum(len(v) for v in all_gt.values())
    print(f"GT 总目标数: {total_gt}\n")

    # ── 加载引擎 ───────────────────────────────────────────────────────────────
    def load_inf(fname, h):
        p = MODELS_DIR / fname
        if not p.exists():
            print(f"  [SKIP] {fname} 不存在，跳过相关配置")
            return None
        sz = p.stat().st_size // 1024 ** 2
        print(f"  ✓ 加载 {fname}  ({sz} MB)  输入高度={h}")
        return Inferencer(str(p), h)

    print("加载 TensorRT 引擎 ...")
    inf375  = load_inf("faster_rcnn_375.engine", 375)
    inf500  = load_inf("faster_rcnn_500.engine", 500)
    inf500i = load_inf("faster_rcnn_500_int8.engine", 500)
    print()

    # ── 阈值函数 ───────────────────────────────────────────────────────────────
    # 固定 0.5 阈值（用于基线和方案1、3）
    thr_fixed    = lambda lb, sc: sc >= 0.50
    # INT8 自适应阈值（方案2）
    thr_adaptive = lambda lb, sc: sc >= ADAPTIVE_THR.get(lb, 0.50)

    # ── 热身（去除冷启动延迟）────────────────────────────────────────────────────
    print("引擎热身中 ...")
    dummy = cv2.imread(str(images[0]))
    for _ in range(5):
        if inf375:  inf375.infer(dummy)
        if inf500:  inf500.infer(dummy)
        if inf500i: inf500i.infer(dummy)
    print("热身完成\n")

    # ── 配置表 ────────────────────────────────────────────────────────────────
    # 每个配置用 name + run_fn(bgr) 描述
    CONFIGS = []

    # 基线
    if inf500:
        _i = inf500
        CONFIGS.append({
            "name": "基线:  FP16_500h  thr=0.5",
            "run":  lambda bgr, i=_i: do_infer(i, bgr, thr_fixed),
        })

    # 方案1：多尺度融合（375h + 500h，均 thr=0.5，NMS 合并）
    if inf375 and inf500:
        _i375, _i500 = inf375, inf500
        CONFIGS.append({
            "name": "方案1: 多尺度融合 (375h+500h) thr=0.5  [↑召回]",
            "run":  lambda bgr, a=_i375, b=_i500: nms_merge([
                        do_infer(a, bgr, thr_fixed),
                        do_infer(b, bgr, thr_fixed),
                    ]),
        })

    # 方案2：INT8_500h + 自适应阈值（速度+难类别召回兼顾）
    if inf500i:
        _i = inf500i
        CONFIGS.append({
            "name": "方案2: INT8_500h  自适应阈值           [↑速度+召回兼顾]",
            "run":  lambda bgr, i=_i: do_infer(i, bgr, thr_adaptive),
        })

    # 方案3：FP16_375h thr=0.5（速度优先）
    if inf375:
        _i = inf375
        CONFIGS.append({
            "name": "方案3: FP16_375h  thr=0.5              [↑速度优先]",
            "run":  lambda bgr, i=_i: do_infer(i, bgr, thr_fixed),
        })

    # ── 逐配置评估 ────────────────────────────────────────────────────────────
    summary = []  # (name, rec, prec, f1, lat_mean, lat_p95, per_cls, overall)

    for cfg in CONFIGS:
        per_image = {}
        latencies = []
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            t0   = time.perf_counter()
            dets = cfg["run"](bgr)
            latencies.append((time.perf_counter() - t0) * 1000)
            per_image[img_path.stem] = dets

        overall, per_cls = compute_metrics(images, all_gt, per_image)
        lat_mean, lat_p95, rec, prec, f1 = print_result(
            cfg["name"], overall, per_cls, latencies)
        summary.append((cfg["name"], rec, prec, f1, lat_mean, lat_p95, per_cls, overall))

    # ── 综合对比汇总表 ────────────────────────────────────────────────────────
    W = 108
    print(f"\n{'═'*W}")
    print(f"  ★ 综合对比表  ({len(images)} 张图 | IoU≥{IOU_THR} | 基线阈值 thr=0.5)")
    print(f"{'═'*W}")
    print(f"  {'配置':48s}{'GT':>5}{'Pred':>7}{'Hit':>6}{'FP':>7}"
          f"{'Recall':>9}{'Prec':>8}{'F1':>7}{'延迟ms':>8}{'P95ms':>8}")
    print("  " + "─" * (W - 2))
    for name, rec, prec, f1, lat, p95, _, overall in summary:
        gt, pred, hit, miss, fp, _, _ = overall
        print(f"  {name:48s}{gt:5d}{pred:7d}{hit:6d}{fp:7d}"
              f"{rec:8.1f}%{prec:7.1f}%{f1:6.1f}%{lat:8.1f}{p95:8.1f}")
    print("  " + "─" * (W - 2))

    # ── 逐类 Recall 对比 ──────────────────────────────────────────────────────
    print(f"\n  逐类 Recall（%）")
    print("  " + "─" * (W - 2))
    cls_hdr = f"  {'配置':48s}" + "".join(f" {EVAL_CLASSES[c]:>11s}" for c in EVAL_CLASSES)
    print(cls_hdr)
    print("  " + "─" * (W - 2))
    for name, rec, prec, f1, lat, p95, per_cls, _ in summary:
        row = f"  {name:48s}"
        for c in EVAL_CLASSES:
            row += f" {per_cls[c][5]:10.1f}%"
        print(row)
    print("  " + "─" * (W - 2))

    # ── 逐类 Precision 对比 ───────────────────────────────────────────────────
    print(f"\n  逐类 Precision（%）")
    print("  " + "─" * (W - 2))
    print(cls_hdr)
    print("  " + "─" * (W - 2))
    for name, rec, prec, f1, lat, p95, per_cls, _ in summary:
        row = f"  {name:48s}"
        for c in EVAL_CLASSES:
            row += f" {per_cls[c][6]:10.1f}%"
        print(row)
    print("  " + "─" * (W - 2))

    # ── 方案优劣总结 ──────────────────────────────────────────────────────────
    if summary:
        best_rec  = max(summary, key=lambda x: x[1])
        best_prec = max(summary, key=lambda x: x[2])
        best_f1   = max(summary, key=lambda x: x[3])
        best_spd  = min(summary, key=lambda x: x[4])
        print(f"\n  ★ 最高 Recall   : {best_rec[0]}  →  {best_rec[1]:.1f}%")
        print(f"  ★ 最高 Precision: {best_prec[0]}  →  {best_prec[2]:.1f}%")
        print(f"  ★ 最高 F1       : {best_f1[0]}  →  {best_f1[3]:.1f}%")
        print(f"  ★ 最快推理      : {best_spd[0]}  →  {best_spd[4]:.1f}ms")

    # ── 关闭引擎 ──────────────────────────────────────────────────────────────
    for inf in [inf375, inf500, inf500i]:
        if inf:
            inf.close()

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out = WS / "test_images/compare_results/eval_precision_summary.json"
    out.parent.mkdir(exist_ok=True)
    save = [
        {
            "config":             n,
            "recall":             r,
            "precision":          p,
            "f1":                 f,
            "lat_mean_ms":        lm,
            "lat_p95_ms":         lp,
            "per_class_recall":    {EVAL_CLASSES[c]: pc[5] for c, pc in pcls.items()},
            "per_class_precision": {EVAL_CLASSES[c]: pc[6] for c, pc in pcls.items()},
            "per_class_gt":        {EVAL_CLASSES[c]: pc[0] for c, pc in pcls.items()},
        }
        for n, r, p, f, lm, lp, pcls, _ in summary
    ]
    json.dump(save, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
