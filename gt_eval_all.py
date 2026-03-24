#!/usr/bin/env python3
"""
gt_eval_all.py
利用 C++ 节点已生成的结果图片（仅用于展示）和比较 JSON，
重新对 100 张 KITTI 图片运行4种配置推理并匹配 GT，输出完整指标表格：
  配置 | GT | Pred | Hit | Miss | FP | Rec% | Prec% | 延迟均值 | P95

GT 类别映射：
  KITTI Pedestrian / Person_sitting → pedestrian(6)
  KITTI Car / Van → car(1)
  KITTI Truck → truck(2)
  KITTI Cyclist → bicycle(8)
  DontCare → 忽略
"""

import os, sys, json, time
from pathlib import Path

import numpy as np
import cv2

# ─────────────────────── 路径 ────────────────────────────────────────────────
WS          = Path("/home/nvidia/ros2_ws")
INSTALL     = WS / "install/faster_rcnn_ros"
NODE_BIN    = INSTALL / "lib/faster_rcnn_ros/faster_rcnn_node"
MODELS_DIR  = INSTALL / "share/faster_rcnn_ros/models"
LABELS_FILE = MODELS_DIR / "labels.txt"
KITTI_IMGS  = WS / "test_images/kitti_100/images"
KITTI_LBLS  = WS / "test_images/kitti_100/labels"
OUT_BASE    = WS / "test_images/compare_results"
ROS_SETUP   = WS / "install/setup.bash"

ENGINE_FP16     = MODELS_DIR / "faster_rcnn_500.engine"
ENGINE_INT8     = MODELS_DIR / "faster_rcnn_500_int8.engine"
ENGINE_FP16_375 = MODELS_DIR / "faster_rcnn_375.engine"

# NOTE: 375×1242 是训练时使用的尺寸，500×1242 需要模型对齐 padding

# KITTI 图片原始尺寸（所有图尺寸一致）
IMG_W, IMG_H = 1224, 370

IOU_THR = 0.5   # 命中判定 IoU 阈值

# ─────────────────────── 类别映射 ────────────────────────────────────────────
CLASS_NAMES = [l.strip() for l in LABELS_FILE.read_text().splitlines() if l.strip()]

# KITTI → NuScenes class_id（只统计这些类别）
KITTI_MAP = {
    "car":            1,
    "van":            1,
    "truck":          2,
    "bus":            3,
    "pedestrian":     6,
    "person_sitting": 6,
    "cyclist":        8,
    "tram":           3,
    "misc":           None,   # 忽略
    "dontcare":       None,   # 忽略
}

# ─────────────────────── IoU ─────────────────────────────────────────────────
def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# ─────────────────────── 解析 KITTI GT ──────────────────────────────────────
def load_gt(img_stem: str):
    """返回 [(cls_id, x1, y1, x2, y2), ...] 过滤 DontCare/misc"""
    lbl = KITTI_LBLS / (img_stem + ".txt")
    result = []
    if not lbl.exists():
        return result
    for line in lbl.read_text().splitlines():
        p = line.strip().split()
        if len(p) < 8:
            continue
        kclass = p[0].lower()
        cls_id = KITTI_MAP.get(kclass)
        if cls_id is None:
            continue
        x1, y1, x2, y2 = float(p[4]), float(p[5]), float(p[6]), float(p[7])
        result.append((cls_id, x1, y1, x2, y2))
    return result

# ─────────────────────── Python TRT 推理（ctypes CUDA）──────────────────────
import ctypes
_cudart = ctypes.cdll.LoadLibrary("libcudart.so")

def _malloc(n):
    p = ctypes.c_void_p()
    assert _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    return p.value

def _free(p): _cudart.cudaFree(ctypes.c_void_p(p))

def _h2d(dst, arr):
    _cudart.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_size_t(arr.nbytes), ctypes.c_int(1))

def _d2h(arr, src):
    _cudart.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src),
                       ctypes.c_size_t(arr.nbytes), ctypes.c_int(2))

import tensorrt as trt

MAX_DET = 2000   # 固定缓冲区最大检测数

class Inferencer:
    """TRT 推理器：与 C++ 节点使用完全相同的预处理 (blobFromImage, scale=1/255, swapRB)"""
    def __init__(self, engine_path, input_h=500, input_w=1242):
        self.INPUT_H = input_h
        self.INPUT_W = input_w
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(
            open(engine_path, "rb").read())
        self._ctx = self._eng.create_execution_context()
        self._ctx.set_input_shape("image", (1, 3, self.INPUT_H, self.INPUT_W))
        # 固定大小缓冲区，动态输出通过 get_tensor_shape 获取实际大小
        self._dinp    = _malloc(3 * self.INPUT_H * self.INPUT_W * 4)
        self._dscores = _malloc(MAX_DET * 4)
        self._dlabels = _malloc(MAX_DET * 4)
        self._dboxes  = _malloc(MAX_DET * 4 * 4)
        self._ctx.set_tensor_address("image",  self._dinp)
        self._ctx.set_tensor_address("scores", self._dscores)
        self._ctx.set_tensor_address("labels", self._dlabels)
        self._ctx.set_tensor_address("boxes",  self._dboxes)

    def infer_raw(self, bgr):
        """与 C++ blobFromImage 完全一致的预处理，返回 (scores, labels, boxes)"""
        blob = cv2.dnn.blobFromImage(bgr, 1.0/255.0, (self.INPUT_W, self.INPUT_H),
                                     swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        _h2d(self._dinp, blob)
        self._ctx.execute_async_v3(0)
        _cudart.cudaDeviceSynchronize()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n <= 0:
            return (np.empty(0, np.float32),
                    np.empty(0, np.int32),
                    np.empty((0, 4), np.float32))
        h_sc = np.empty(n,      dtype=np.float32); _d2h(h_sc, self._dscores)
        h_lb = np.empty(n,      dtype=np.int32);   _d2h(h_lb, self._dlabels)
        h_bx = np.empty(n * 4, dtype=np.float32); _d2h(h_bx, self._dboxes)
        return h_sc, h_lb, h_bx.reshape(n, 4)

    def close(self):
        _free(self._dinp); _free(self._dscores); _free(self._dlabels); _free(self._dboxes)


def run_inference_with_boxes(engine_path, threshold, images, input_h=500, input_w=1242):
    """
    对每张图推理，返回:
      per_image: {stem: [(cls_id, x1,y1,x2,y2, score), ...]}
      latencies: [ms, ...]
    """
    inf = Inferencer(engine_path, input_h=input_h, input_w=input_w)
    # warmup
    dummy = cv2.imread(str(images[0]))
    for _ in range(5): inf.infer_raw(dummy)

    per_image = {}
    latencies = []

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None: continue
        oh, ow = bgr.shape[:2]
        sx, sy = ow / inf.INPUT_W, oh / inf.INPUT_H

        t0 = time.perf_counter()
        scores, labels, boxes = inf.infer_raw(bgr)
        latencies.append((time.perf_counter() - t0) * 1000)

        stem = img_path.stem
        dets = []
        for i in range(len(scores)):
            sc = float(scores[i])
            if sc < threshold: continue
            lb = int(labels[i])
            x1 = float(boxes[i, 0] * sx); y1 = float(boxes[i, 1] * sy)
            x2 = float(boxes[i, 2] * sx); y2 = float(boxes[i, 3] * sy)
            if x2 <= x1 or y2 <= y1: continue
            dets.append((lb, x1, y1, x2, y2, sc))
        per_image[stem] = dets

    inf.close()
    return per_image, latencies


# ─────────────────────── GT 匹配（IoU）────────────────────────────────────────
# NuScenes class_id 对应 KITTI 类别的宽泛映射（允许跨标签命中）
COMPAT = {
    1: {1},       # car → car
    2: {1, 2},    # truck → car/truck
    3: {2, 3},    # bus → truck/bus
    6: {6},       # pedestrian → pedestrian
    7: {7, 8},    # motorcycle → motorcycle/bicycle
    8: {7, 8},    # bicycle → motorcycle/bicycle
}

def match_gt(gt_list, pred_list, iou_thr=IOU_THR):
    """
    gt_list:   [(cls_id, x1,y1,x2,y2), ...]
    pred_list: [(cls_id, x1,y1,x2,y2, score), ...]
    返回 (hit, miss, fp)
    """
    matched_gt  = set()
    matched_pred = set()

    # 按 score 降序
    preds_sorted = sorted(enumerate(pred_list), key=lambda x: -x[1][5])

    for pi, pred in preds_sorted:
        best_iou, best_gi = 0.0, -1
        pcls = pred[0]
        pcls_compat = COMPAT.get(pcls, {pcls})
        pb = pred[1:5]
        for gi, gt in enumerate(gt_list):
            if gi in matched_gt: continue
            gcls = gt[0]
            if gcls not in pcls_compat and pcls not in COMPAT.get(gcls, {gcls}):
                continue
            v = iou(pb, gt[1:5])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_thr and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    hit  = len(matched_gt)
    miss = len(gt_list) - hit
    fp   = len(pred_list) - len(matched_pred)
    return hit, miss, fp


# ─────────────────────── 主流程 ──────────────────────────────────────────────
# (名称, 引擎路径, 置信度阈值, 输入高度, 输入宽度)
# 训练尺寸: 375×1242  →  FP16_375h 是与训练对齐的配置
CONFIGS = [
    ("FP16_375h(训练尺寸) thr=0.3", str(ENGINE_FP16_375), 0.30, 375, 1242),
    ("FP16_375h(训练尺寸) thr=0.5", str(ENGINE_FP16_375), 0.50, 375, 1242),
    ("FP16_500h            thr=0.3", str(ENGINE_FP16),     0.30, 500, 1242),
    ("FP16_500h            thr=0.5", str(ENGINE_FP16),     0.50, 500, 1242),
    ("INT8_500h            thr=0.3", str(ENGINE_INT8),     0.30, 500, 1242),
    ("INT8_500h            thr=0.5", str(ENGINE_INT8),     0.50, 500, 1242),
]

def main():
    images = sorted(KITTI_IMGS.glob("*.png"))[:100]
    print(f"图片: {len(images)} 张  |  GT 标注: {KITTI_LBLS}")

    # 预加载所有 GT
    all_gt = {}
    for img in images:
        all_gt[img.stem] = load_gt(img.stem)

    total_gt_objs = sum(len(v) for v in all_gt.values())
    print(f"GT 总目标数（过滤 DontCare）: {total_gt_objs}")

    results = []

    for cfg_name, engine_path, threshold, inp_h, inp_w in CONFIGS:
        if not Path(engine_path).exists():
            print(f"[SKIP] {cfg_name}: 引擎不存在")
            continue
        prec_tag = "fp16" if "FP16" in cfg_name else "int8"
        thr_tag  = f"{inp_h}h_{str(threshold).replace('.', 'p')}"
        log_tag  = f"{prec_tag}_thr{thr_tag}"

        print(f"\n{'─'*60}")
        print(f"  ▶ {cfg_name}  [{Path(engine_path).name}]")

        per_image, latencies = run_inference_with_boxes(engine_path, threshold, images, inp_h, inp_w)

        # 汇总指标
        tot_gt = tot_pred = tot_hit = tot_miss = tot_fp = 0
        for img in images:
            stem = img.stem
            gt   = all_gt.get(stem, [])
            pred = per_image.get(stem, [])
            hit, miss, fp = match_gt(gt, pred, IOU_THR)
            tot_gt   += len(gt)
            tot_pred += len(pred)
            tot_hit  += hit
            tot_miss += miss
            tot_fp   += fp

        rec  = tot_hit / tot_gt  * 100 if tot_gt  > 0 else 0.0
        prec = tot_hit / tot_pred * 100 if tot_pred > 0 else 0.0
        lat_mean   = float(np.mean(latencies))
        lat_median = float(np.median(latencies))
        lat_p95    = float(np.percentile(latencies, 95))

        engine_mb = Path(engine_path).stat().st_size // 1024**2

        results.append({
            "cfg"       : cfg_name,
            "engine_mb" : engine_mb,
            "threshold" : threshold,
            "GT"        : tot_gt,
            "Pred"      : tot_pred,
            "Hit"       : tot_hit,
            "Miss"      : tot_miss,
            "FP"        : tot_fp,
            "Rec%"      : rec,
            "Prec%"     : prec,
            "lat_mean"  : lat_mean,
            "lat_median": lat_median,
            "lat_p95"   : lat_p95,
        })

        print(f"  GT={tot_gt}  Pred={tot_pred}  Hit={tot_hit}  Miss={tot_miss}  FP={tot_fp}")
        print(f"  Recall={rec:.1f}%  Precision={prec:.1f}%  |  "
              f"延迟 mean={lat_mean:.1f}ms  median={lat_median:.1f}ms  P95={lat_p95:.1f}ms")

    # ── 汇总表格 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  ★ 综合指标对比表（KITTI 100 张图片 | IoU≥{:.1f}）".format(IOU_THR))
    print(f"{'='*100}")

    COL = [
        ("配置",       20, "cfg",        "s"),
        ("引擎",        8, "engine_mb",  "d"),
        ("GT",          6, "GT",         "d"),
        ("Pred",        6, "Pred",       "d"),
        ("Hit",         6, "Hit",        "d"),
        ("Miss",        6, "Miss",       "d"),
        ("FP",          6, "FP",         "d"),
        ("Rec%",        7, "Rec%",       ".1f"),
        ("Prec%",       7, "Prec%",      ".1f"),
        ("延迟均值",   10, "lat_mean",   ".1f"),
        ("延迟中位",   10, "lat_median", ".1f"),
        ("延迟P95",     9, "lat_p95",    ".1f"),
    ]

    hdr = ""
    sep = ""
    for col_name, width, _, _ in COL:
        hdr += f"{col_name:>{width}} "
        sep += "-" * width + " "
    print(hdr)
    print(sep)

    for r in results:
        row = ""
        for col_name, width, key, fmt in COL:
            v = r[key]
            if fmt == "s":
                row += f"{v:>{width}} "
            elif fmt == "d":
                row += f"{v:>{width}d} "
            else:
                row += f"{v:>{width}.{fmt.replace('.','').replace('f','')}f} "
        print(row)

    print(sep)

    # ── 精度提升对比 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ▷ 关键对比分析:")
    rm = {r["cfg"]: r for r in results}

    fp16_05 = rm.get("FP16 + thr=0.5")
    fp16_25 = rm.get("FP16 + thr=0.25")
    int8_05 = rm.get("INT8 + thr=0.5")
    int8_25 = rm.get("INT8 + thr=0.25")

    if fp16_05 and fp16_25:
        drec  = fp16_25["Rec%"]  - fp16_05["Rec%"]
        dfp   = fp16_25["FP"]    - fp16_05["FP"]
        dlat  = fp16_25["lat_mean"] - fp16_05["lat_mean"]
        print(f"  FP16: 0.5→0.25  Recall {fp16_05['Rec%']:.1f}%→{fp16_25['Rec%']:.1f}% "
              f"({drec:+.1f}%)  FP {fp16_05['FP']}→{fp16_25['FP']} ({dfp:+d})  "
              f"延迟差 {dlat:+.1f}ms")

    if int8_05 and int8_25:
        drec  = int8_25["Rec%"]  - int8_05["Rec%"]
        dfp   = int8_25["FP"]    - int8_05["FP"]
        dlat  = int8_25["lat_mean"] - int8_05["lat_mean"]
        print(f"  INT8: 0.5→0.25  Recall {int8_05['Rec%']:.1f}%→{int8_25['Rec%']:.1f}% "
              f"({drec:+.1f}%)  FP {int8_05['FP']}→{int8_25['FP']} ({dfp:+d})  "
              f"延迟差 {dlat:+.1f}ms")

    if fp16_05 and int8_05:
        drec = int8_05["Rec%"] - fp16_05["Rec%"]
        dlat = int8_05["lat_mean"] - fp16_05["lat_mean"]
        dsz  = int8_05["engine_mb"] - fp16_05["engine_mb"]
        print(f"  同阈值0.5: FP16→INT8  Recall {drec:+.1f}%  延迟 {dlat:+.1f}ms  "
              f"引擎 {dsz:+d}MB")

    if fp16_25 and int8_25:
        drec = int8_25["Rec%"] - fp16_25["Rec%"]
        dlat = int8_25["lat_mean"] - fp16_25["lat_mean"]
        print(f"  同阈值0.25: FP16→INT8  Recall {drec:+.1f}%  延迟 {dlat:+.1f}ms")

    # ── 保存 JSON ─────────────────────────────────────────────────────────
    out = OUT_BASE / "gt_eval_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n汇总结果已保存: {out}")


if __name__ == "__main__":
    main()
