#!/usr/bin/env python3
"""
optimize_eval.py
验证三种正向优化方案（使用已有引擎，无需重建）：
  方案1: 理论召回上限 — 阈值=0.05，看模型天花板
  方案2: 多尺度合并推理 — 375+500 TRT 结果合并+类间IoU NMS
  方案3: 按类别分组阈值 — 车辆0.3，行人/骑手0.15

GT: KITTI 100张  IoU≥0.5  (与 gt_eval_all.py 保持一致)
"""

import os, sys, time
from pathlib import Path
import numpy as np
import cv2
import ctypes

_cudart = ctypes.cdll.LoadLibrary("libcudart.so")
def _malloc(n): p=ctypes.c_void_p(); assert _cudart.cudaMalloc(ctypes.byref(p),ctypes.c_size_t(n))==0; return p.value
def _free(p): _cudart.cudaFree(ctypes.c_void_p(p))
def _h2d(dst,arr): _cudart.cudaMemcpy(ctypes.c_void_p(dst),arr.ctypes.data_as(ctypes.c_void_p),ctypes.c_size_t(arr.nbytes),ctypes.c_int(1))
def _d2h(arr,src): _cudart.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p),ctypes.c_void_p(src),ctypes.c_size_t(arr.nbytes),ctypes.c_int(2))

import tensorrt as trt

MAX_DET = 2000

WS          = Path("/home/nvidia/ros2_ws")
MODELS_DIR  = WS / "install/faster_rcnn_ros/share/faster_rcnn_ros/models"
KITTI_IMGS  = WS / "test_images/kitti_100/images"
KITTI_LBLS  = WS / "test_images/kitti_100/labels"

ENGINE_375  = MODELS_DIR / "faster_rcnn_375.engine"
ENGINE_500  = MODELS_DIR / "faster_rcnn_500.engine"
ENGINE_700  = MODELS_DIR / "faster_rcnn_700.engine"
ENGINE_INT8 = MODELS_DIR / "faster_rcnn_500_int8.engine"

IOU_THR = 0.5

KITTI_MAP = {
    "car":1, "van":1, "truck":2, "bus":3,
    "pedestrian":6, "person_sitting":6, "cyclist":8,
    "tram":3, "misc":None, "dontcare":None,
}

# 类别兼容映射（宽泛匹配）
COMPAT = {1:{1},2:{1,2},3:{2,3},6:{6},7:{7,8},8:{7,8}}

# ─── 推理器 ────────────────────────────────────────────────────────────────────
class Inferencer:
    def __init__(self, engine_path, input_h=500, input_w=1242):
        self.INPUT_H, self.INPUT_W = input_h, input_w
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path,"rb").read())
        self._ctx = self._eng.create_execution_context()
        self._ctx.set_input_shape("image", (1,3,input_h,input_w))
        self._dinp    = _malloc(3*input_h*input_w*4)
        self._dscores = _malloc(MAX_DET*4)
        self._dlabels = _malloc(MAX_DET*4)
        self._dboxes  = _malloc(MAX_DET*4*4)
        self._ctx.set_tensor_address("image",  self._dinp)
        self._ctx.set_tensor_address("scores", self._dscores)
        self._ctx.set_tensor_address("labels", self._dlabels)
        self._ctx.set_tensor_address("boxes",  self._dboxes)

    def infer_raw(self, bgr):
        blob = cv2.dnn.blobFromImage(bgr, 1.0/255.0, (self.INPUT_W, self.INPUT_H), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        _h2d(self._dinp, blob)
        self._ctx.execute_async_v3(0); _cudart.cudaDeviceSynchronize()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n <= 0: return np.empty(0,np.float32), np.empty(0,np.int32), np.empty((0,4),np.float32)
        h_sc = np.empty(n,dtype=np.float32); _d2h(h_sc,self._dscores)
        h_lb = np.empty(n,dtype=np.int32);   _d2h(h_lb,self._dlabels)
        h_bx = np.empty(n*4,dtype=np.float32); _d2h(h_bx,self._dboxes)
        return h_sc, h_lb, h_bx.reshape(n,4)

    def close(self): _free(self._dinp); _free(self._dscores); _free(self._dlabels); _free(self._dboxes)


# ─── GT 加载 ──────────────────────────────────────────────────────────────────
def load_gt(stem):
    lbl = KITTI_LBLS / (stem+".txt")
    result = []
    if not lbl.exists(): return result
    for line in lbl.read_text().splitlines():
        p = line.strip().split()
        if len(p) < 8: continue
        cls_id = KITTI_MAP.get(p[0].lower())
        if cls_id is None: continue
        result.append((cls_id, float(p[4]), float(p[5]), float(p[6]), float(p[7])))
    return result

def iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def match_gt(gt_list, pred_list):
    matched_gt=set(); matched_pred=set()
    for pi, pred in sorted(enumerate(pred_list), key=lambda x:-x[1][5]):
        best_iou, best_gi = 0.0, -1
        pcls_compat = COMPAT.get(pred[0], {pred[0]})
        pb = pred[1:5]
        for gi, gt in enumerate(gt_list):
            if gi in matched_gt: continue
            gcls = gt[0]
            if gcls not in pcls_compat and pred[0] not in COMPAT.get(gcls,{gcls}): continue
            v = iou(pb, gt[1:5])
            if v > best_iou: best_iou, best_gi = v, gi
        if best_iou >= IOU_THR and best_gi >= 0:
            matched_gt.add(best_gi); matched_pred.add(pi)
    return len(matched_gt), len(gt_list)-len(matched_gt), len(pred_list)-len(matched_pred)


# ─── NMS（用于多尺度合并去重）─────────────────────────────────────────────────
def nms_boxes(dets, iou_thr=0.5):
    """dets: [(cls, x1,y1,x2,y2,score), ...]，返回保留后的列表"""
    if not dets: return []
    dets = sorted(dets, key=lambda x:-x[5])
    keep = []
    suppressed = set()
    for i, d in enumerate(dets):
        if i in suppressed: continue
        keep.append(d)
        for j in range(i+1, len(dets)):
            if j in suppressed: continue
            if iou(d[1:5], dets[j][1:5]) > iou_thr:
                suppressed.add(j)
    return keep


# ─── 推理并返回逐图检测结果（支持自定义 threshold 函数）────────────────────────
def infer_all(inf: Inferencer, images, thr_fn):
    """
    thr_fn(label_id, score) → True 保留
    返回 (per_image, latencies)
    """
    dummy = cv2.imread(str(images[0]))
    for _ in range(3): inf.infer_raw(dummy)

    per_image = {}; latencies = []
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None: continue
        oh, ow = bgr.shape[:2]
        sx, sy = ow/inf.INPUT_W, oh/inf.INPUT_H
        t0 = time.perf_counter()
        scores, labels, boxes = inf.infer_raw(bgr)
        latencies.append((time.perf_counter()-t0)*1000)
        dets = []
        for i in range(len(scores)):
            sc=float(scores[i]); lb=int(labels[i])
            if not thr_fn(lb, sc): continue
            x1=float(boxes[i,0]*sx); y1=float(boxes[i,1]*sy)
            x2=float(boxes[i,2]*sx); y2=float(boxes[i,3]*sy)
            if x2<=x1 or y2<=y1: continue
            dets.append((lb,x1,y1,x2,y2,sc))
        per_image[img_path.stem] = dets
    return per_image, latencies


def eval_perimage(images, all_gt, per_image):
    tot_gt=tot_pred=tot_hit=tot_miss=tot_fp=0
    for img in images:
        gt=all_gt.get(img.stem,[]); pred=per_image.get(img.stem,[])
        hit,miss,fp=match_gt(gt,pred)
        tot_gt+=len(gt); tot_pred+=len(pred)
        tot_hit+=hit; tot_miss+=miss; tot_fp+=fp
    rec  = tot_hit/tot_gt*100  if tot_gt  else 0
    prec = tot_hit/tot_pred*100 if tot_pred else 0
    return tot_gt, tot_pred, tot_hit, tot_miss, tot_fp, rec, prec


def print_result(name, gt, pred, hit, miss, fp, rec, prec, lat_mean, lat_p95):
    print(f"  {'─'*60}")
    print(f"  ▶ {name}")
    print(f"    GT={gt}  Pred={pred}  Hit={hit}  Miss={miss}  FP={fp}")
    print(f"    Recall={rec:.1f}%  Precision={prec:.1f}%  |  "
          f"延迟 mean={lat_mean:.1f}ms  P95={lat_p95:.1f}ms")


def main():
    images = sorted(KITTI_IMGS.glob("*.png"))[:100]
    print(f"图片: {len(images)} | GT: {KITTI_LBLS}")
    all_gt = {img.stem: load_gt(img.stem) for img in images}
    total_gt = sum(len(v) for v in all_gt.values())
    print(f"GT 总目标: {total_gt}\n")

    results = []

    # ══════════════════════════════════════════════════════════════════════════
    # 方案1: 阈值=0.05 → 理论召回上限（用 500 FP16 引擎）
    # ══════════════════════════════════════════════════════════════════════════
    print("╔══ 方案1: 理论召回上限（阈值=0.05，500h FP16）══╗")
    inf500 = Inferencer(str(ENGINE_500), 500, 1242)
    per, lats = infer_all(inf500, images, lambda lb,sc: sc>=0.05)
    inf500.close()
    gt,pred,hit,miss,fp,rec,prec = eval_perimage(images, all_gt, per)
    print_result("FP16_500h  thr=0.05 [上限]", gt,pred,hit,miss,fp,rec,prec,
                 np.mean(lats), np.percentile(lats,95))
    results.append(("FP16_500h thr=0.05[上限]", gt,pred,hit,miss,fp,rec,prec,np.mean(lats),np.percentile(lats,95)))

    # ══════════════════════════════════════════════════════════════════════════
    # 方案2: 多尺度合并推理（375 + 500，合并后 NMS 去重，thr=0.25）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n╔══ 方案2: 多尺度合并推理（375h+500h，thr=0.25，NMS_iou=0.5）══╗")
    inf375 = Inferencer(str(ENGINE_375), 375, 1242)
    inf500 = Inferencer(str(ENGINE_500), 500, 1242)
    # warmup
    dummy = cv2.imread(str(images[0]))
    for _ in range(3): inf375.infer_raw(dummy); inf500.infer_raw(dummy)

    per_ms = {}; lats_ms = []
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None: continue
        oh, ow = bgr.shape[:2]
        t0 = time.perf_counter()
        # 375 推理
        sc1,lb1,bx1 = inf375.infer_raw(bgr)
        sx1,sy1 = ow/375, oh/375
        d1 = [(int(lb1[i]), float(bx1[i,0]*sx1), float(bx1[i,1]*sy1),
               float(bx1[i,2]*sx1), float(bx1[i,3]*sy1), float(sc1[i]))
              for i in range(len(sc1)) if sc1[i]>=0.25]
        # 500 推理
        sc2,lb2,bx2 = inf500.infer_raw(bgr)
        sx2,sy2 = ow/1242, oh/500
        d2 = [(int(lb2[i]), float(bx2[i,0]*sx2), float(bx2[i,1]*sy2),
               float(bx2[i,2]*sx2), float(bx2[i,3]*sy2), float(sc2[i]))
              for i in range(len(sc2)) if sc2[i]>=0.25]
        # 合并 + 类内 NMS
        merged = d1 + d2
        kept = nms_boxes(merged, iou_thr=0.5)
        lats_ms.append((time.perf_counter()-t0)*1000)
        per_ms[img_path.stem] = kept

    inf375.close(); inf500.close()
    gt,pred,hit,miss,fp,rec,prec = eval_perimage(images, all_gt, per_ms)
    print_result("多尺度 375+500  thr=0.25  NMS合并", gt,pred,hit,miss,fp,rec,prec,
                 np.mean(lats_ms), np.percentile(lats_ms,95))
    results.append(("多尺度375+500 thr=0.25", gt,pred,hit,miss,fp,rec,prec,np.mean(lats_ms),np.percentile(lats_ms,95)))

    # ══════════════════════════════════════════════════════════════════════════
    # 方案3a: 按类别分组阈值（500h FP16）
    #   车辆(1,2,3): thr=0.30   行人(6)/骑手(7,8): thr=0.15
    # ══════════════════════════════════════════════════════════════════════════
    PED_CLASSES = {6, 7, 8}
    VEH_THR, PED_THR = 0.30, 0.15
    def per_class_thr(lb, sc, veh=VEH_THR, ped=PED_THR):
        return sc >= (ped if lb in PED_CLASSES else veh)

    print(f"\n╔══ 方案3a: 按类别阈值（500h FP16）车辆thr={VEH_THR} 行人thr={PED_THR}  ══╗")
    inf500 = Inferencer(str(ENGINE_500), 500, 1242)
    per, lats = infer_all(inf500, images, per_class_thr)
    inf500.close()
    gt,pred,hit,miss,fp,rec,prec = eval_perimage(images, all_gt, per)
    print_result(f"FP16_500h  veh_thr={VEH_THR}  ped_thr={PED_THR}", gt,pred,hit,miss,fp,rec,prec,
                 np.mean(lats), np.percentile(lats,95))
    results.append((f"按类别阈值 500h veh={VEH_THR}/ped={PED_THR}", gt,pred,hit,miss,fp,rec,prec,np.mean(lats),np.percentile(lats,95)))

    # ══════════════════════════════════════════════════════════════════════════
    # 方案3b: 按类别分组阈值 + INT8（更快推理）
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n╔══ 方案3b: 按类别阈值（500h INT8）车辆thr={VEH_THR} 行人thr={PED_THR}  ══╗")
    inf_i8 = Inferencer(str(ENGINE_INT8), 500, 1242)
    per, lats = infer_all(inf_i8, images, per_class_thr)
    inf_i8.close()
    gt,pred,hit,miss,fp,rec,prec = eval_perimage(images, all_gt, per)
    print_result(f"INT8_500h  veh_thr={VEH_THR}  ped_thr={PED_THR}", gt,pred,hit,miss,fp,rec,prec,
                 np.mean(lats), np.percentile(lats,95))
    results.append((f"按类别阈值 INT8 500h veh={VEH_THR}/ped={PED_THR}", gt,pred,hit,miss,fp,rec,prec,np.mean(lats),np.percentile(lats,95)))

    # ══════════════════════════════════════════════════════════════════════════
    # 方案4: 700h 引擎（若已构建完成）— 与训练分辨率匹配
    # ══════════════════════════════════════════════════════════════════════════
    if ENGINE_700.exists():
        print(f"\n╔══ 方案4: 训练匹配分辨率（700h FP16）thr=0.25  ══╗")
        inf700 = Inferencer(str(ENGINE_700), 700, 1242)
        per, lats = infer_all(inf700, images, lambda lb,sc: sc>=0.25)
        inf700.close()
        gt,pred,hit,miss,fp,rec,prec = eval_perimage(images, all_gt, per)
        print_result("FP16_700h  thr=0.25 [训练分辨率]", gt,pred,hit,miss,fp,rec,prec,
                     np.mean(lats), np.percentile(lats,95))
        results.append(("FP16_700h thr=0.25[训练分辨率]", gt,pred,hit,miss,fp,rec,prec,np.mean(lats),np.percentile(lats,95)))
    else:
        print(f"\n[跳过] 700h 引擎尚未构建完成: {ENGINE_700}")

    # ══════════════════════════════════════════════════════════════════════════
    # 综合对比表
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  ★ 优化方案对比表（IoU≥0.5，KITTI 100张）")
    print(f"{'='*90}")
    hdr = f"{'方案':35s} {'GT':>5} {'Pred':>6} {'Hit':>5} {'FP':>6} {'Recall':>8} {'Prec':>8} {'延迟ms':>8} {'P95':>7}"
    print(hdr)
    print("-"*90)
    for r in results:
        name,gt,pred,hit,miss,fp,rec,prec,lat,p95 = r
        print(f"  {name:33s} {gt:5d} {pred:6d} {hit:5d} {fp:6d} {rec:7.1f}% {prec:7.1f}% {lat:8.1f} {p95:7.1f}")
    print("-"*90)

    # 基准对比行（来自 gt_eval_all.py 已知结果）
    print()
    print("  [基准] FP16_375h thr=0.3:  Recall=63.2%  Prec=43.5%  延迟62ms")
    print("  [基准] FP16_500h thr=0.3:  Recall=67.1%  Prec=40.1%  延迟75ms")
    print("  [基准] INT8_500h thr=0.3:  Recall=68.8%  Prec=37.1%  延迟66ms")
    print()
    best_rec = max(results, key=lambda x: x[6])
    best_prec = max(results, key=lambda x: x[7])
    best_bal = max(results, key=lambda x: (x[6]+x[7])/2)
    print(f"  ★ 最高 Recall:    {best_rec[0]}  →  {best_rec[6]:.1f}%")
    print(f"  ★ 最高 Precision: {best_prec[0]}  →  {best_prec[7]:.1f}%")
    print(f"  ★ 最佳 F1 均衡:   {best_bal[0]}  →  F1≈{2*best_bal[6]*best_bal[7]/(best_bal[6]+best_bal[7]):.1f}%")


if __name__ == '__main__':
    main()
