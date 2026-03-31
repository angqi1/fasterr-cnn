#!/usr/bin/env python3
"""
eval_yolo_sslad.py - 原生TRT推理YOLOv8n + SSLAD-2D GT精度评估
"""
import time, os
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import ctypes
import tensorrt as trt

WS         = Path("/home/nvidia/ros2_ws")
SSLAD_IMGS = WS / "test_images/sslad_300/images"
SSLAD_LBLS = WS / "test_images/sslad_300/labels"
YOLO_ENGINE = str(WS / "yolov8n_trt.engine")
INPUT_SZ   = 640
IOU_THR    = 0.5
NMS_IOU    = 0.45

SSLAD_TO_UID = {"pedestrian":"pedestrian","cyclist":"cyclist","car":"car","truck":"truck","tram":"tram"}
COCO_TO_UID = {0:"pedestrian",1:"cyclist",2:"car",3:"cyclist",5:"tram",7:"truck"}
COMPAT = {"car":{"car"},"truck":{"car","truck"},"tram":{"truck","tram"},"pedestrian":{"pedestrian"},"cyclist":{"cyclist"}}

_cudart = ctypes.cdll.LoadLibrary("libcudart.so")
def _malloc(n):
    p = ctypes.c_void_p(); assert _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0; return p.value
def _free(p): _cudart.cudaFree(ctypes.c_void_p(p))
def _h2d(dst, arr): _cudart.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(arr.nbytes), ctypes.c_int(1))
def _d2h(arr, src): _cudart.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src), ctypes.c_size_t(arr.nbytes), ctypes.c_int(2))

def iou_calc(a, b):
    ix1=max(a[0],b[0]);iy1=max(a[1],b[1]);ix2=min(a[2],b[2]);iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def load_gt(stem):
    lbl = SSLAD_LBLS / (stem + ".txt"); result = []
    if not lbl.exists(): return result
    for line in lbl.read_text().splitlines():
        p = line.strip().split()
        if len(p) < 8: continue
        uid = SSLAD_TO_UID.get(p[0].lower())
        if uid is None: continue
        result.append((uid, float(p[4]), float(p[5]), float(p[6]), float(p[7])))
    return result

def nms_boxes(dets, iou_thr=NMS_IOU):
    if not dets: return dets
    dets_sorted = sorted(dets, key=lambda x: -x[5]); keep = []
    for d in dets_sorted:
        discard = False
        for k in keep:
            if iou_calc(d[1:5], k[1:5]) > iou_thr: discard = True; break
        if not discard: keep.append(d)
    return keep

def letterbox(bgr, sz=640):
    h, w = bgr.shape[:2]; r = min(sz/h, sz/w)
    nw, nh = int(w*r), int(h*r)
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
    px, py = (sz-nw)//2, (sz-nh)//2
    canvas[py:py+nh, px:px+nw] = resized
    blob = canvas[:,:,::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2,0,1)[np.newaxis])
    return blob, (r, px, py)

def postprocess(output, conf_thr, oh, ow, r, px, py):
    out = output[0].T  # (8400, 84)
    cx, cy, bw, bh = out[:,0], out[:,1], out[:,2], out[:,3]
    class_scores = out[:, 4:]
    cls_ids = np.argmax(class_scores, axis=1)
    max_scores = class_scores[np.arange(len(cls_ids)), cls_ids]
    mask = max_scores > conf_thr
    cx,cy,bw,bh = cx[mask],cy[mask],bw[mask],bh[mask]
    cls_ids, max_scores = cls_ids[mask], max_scores[mask]
    x1=cx-bw/2; y1=cy-bh/2; x2=cx+bw/2; y2=cy+bh/2
    x1=(x1-px)/r; y1=(y1-py)/r; x2=(x2-px)/r; y2=(y2-py)/r
    x1=np.clip(x1,0,ow); y1=np.clip(y1,0,oh); x2=np.clip(x2,0,ow); y2=np.clip(y2,0,oh)
    dets = []
    for i in range(len(cls_ids)):
        uid = COCO_TO_UID.get(int(cls_ids[i]))
        if uid is None: continue
        if x2[i]<=x1[i] or y2[i]<=y1[i]: continue
        dets.append((uid, float(x1[i]),float(y1[i]),float(x2[i]),float(y2[i]),float(max_scores[i])))
    return nms_boxes(dets)

def match_gt(gt_list, pred_list, iou_thr):
    matched_gt=set(); matched_pred=set(); cls_hits=Counter()
    preds_sorted = sorted(enumerate(pred_list), key=lambda x:-x[1][5])
    for pi, pred in preds_sorted:
        best_iou,best_gi=0.0,-1; pcls=pred[0]; pcls_compat=COMPAT.get(pcls,{pcls}); pb=pred[1:5]
        for gi,gt in enumerate(gt_list):
            if gi in matched_gt: continue
            gcls=gt[0]
            if gcls not in pcls_compat and pcls not in COMPAT.get(gcls,{gcls}): continue
            v=iou_calc(pb,gt[1:5])
            if v>best_iou: best_iou=v; best_gi=gi
        if best_iou>=iou_thr and best_gi>=0:
            matched_gt.add(best_gi); matched_pred.add(pi); cls_hits[gt_list[best_gi][0]]+=1
    return len(matched_gt), len(gt_list)-len(matched_gt), len(pred_list)-len(matched_pred), cls_hits

def main():
    images = sorted(SSLAD_IMGS.glob("*.jpg"))[:300]
    print(f"图片: {len(images)} 张")
    all_gt = {}
    for img in images: all_gt[img.stem] = load_gt(img.stem)
    total_gt = sum(len(v) for v in all_gt.values())
    print(f"GT 总目标数: {total_gt}")

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    with open(YOLO_ENGINE, "rb") as f:
        eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()

    n_io = eng.num_bindings
    print(f"引擎绑定数: {n_io}")
    for i in range(n_io):
        name = eng.get_binding_name(i)
        shape = eng.get_binding_shape(i)
        is_input = eng.binding_is_input(i)
        print(f"  [{i}] {'INPUT' if is_input else 'OUTPUT'} {name}: {list(shape)}")

    # 获取实际输出shape
    ctx.set_binding_shape(0, (1, 3, INPUT_SZ, INPUT_SZ))
    out_shape = tuple(ctx.get_binding_shape(1))
    print(f"输出tensor shape: {out_shape}")
    out_numel = 1
    for d in out_shape: out_numel *= d

    d_inp = _malloc(1*3*INPUT_SZ*INPUT_SZ*4)
    d_out = _malloc(out_numel*4)
    bindings = (ctypes.c_void_p * n_io)(d_inp, d_out)

    # Warmup
    dummy = cv2.imread(str(images[0]))
    blob, _ = letterbox(dummy, INPUT_SZ)
    for _ in range(15):
        _h2d(d_inp, blob); ctx.execute_v2(bindings); _cudart.cudaDeviceSynchronize()
    print("预热完成\n")

    for threshold in [0.25, 0.3, 0.5]:
        latencies = []; tot_gt=tot_pred=tot_hit=tot_miss=tot_fp=0
        cls_gt_cnt=Counter(); cls_hit_cnt=Counter()
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            oh, ow = bgr.shape[:2]
            blob, (r, px, py) = letterbox(bgr, INPUT_SZ)
            _h2d(d_inp, blob)
            t0 = time.perf_counter()
            ctx.execute_v2(bindings)
            _cudart.cudaDeviceSynchronize()
            latencies.append((time.perf_counter()-t0)*1000)
            h_out = np.empty(out_shape, dtype=np.float32)
            _d2h(h_out, d_out)
            dets = postprocess(h_out, threshold, oh, ow, r, px, py)
            gt = all_gt.get(img_path.stem, [])
            for g in gt: cls_gt_cnt[g[0]] += 1
            hit,miss,fp,cls_hits = match_gt(gt, dets, IOU_THR)
            for k,v in cls_hits.items(): cls_hit_cnt[k] += v
            tot_gt+=len(gt); tot_pred+=len(dets); tot_hit+=hit; tot_miss+=miss; tot_fp+=fp

        rec = tot_hit/tot_gt*100 if tot_gt>0 else 0
        prec = tot_hit/tot_pred*100 if tot_pred>0 else 0
        lat = np.array(latencies)
        print(f"=== YOLOv8n TRT FP16 640 | SSLAD 300张 | thr={threshold} | IoU≥{IOU_THR} ===")
        print(f"  GT={tot_gt}  Pred={tot_pred}  Hit={tot_hit}  Miss={tot_miss}  FP={tot_fp}")
        print(f"  Recall={rec:.1f}%  Precision={prec:.1f}%")
        print(f"  延迟: mean={lat.mean():.1f}ms  median={np.median(lat):.1f}ms  min={lat.min():.1f}ms  max={lat.max():.1f}ms  P90={np.percentile(lat,90):.1f}ms  P95={np.percentile(lat,95):.1f}ms")
        print(f"  --- 按类别 Recall ---")
        for cls in sorted(cls_gt_cnt.keys()):
            g=cls_gt_cnt[cls]; h=cls_hit_cnt.get(cls,0)
            print(f"    {cls:>12}: {h}/{g} = {h/g*100:.1f}%")
        print()

    _free(d_inp); _free(d_out)
    print("✅ 评估完成")

if __name__ == "__main__":
    main()
