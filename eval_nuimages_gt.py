#!/usr/bin/env python3
"""
eval_nuimages_gt.py — nuImages mini 数据集完整评估
  Part 1: GT 精确评估 (8张keyframe, 带2D bbox标注)
           → 各类别 Recall / Precision / F1 @ IoU≥0.5
  Part 2: 检测覆盖统计 (96张mini sweeps + 500张连续sweep)
           → 各类别每帧检测数 / 置信度P50

用法:
  python3 eval_nuimages_gt.py
  python3 eval_nuimages_gt.py --thr 0.5
"""

import argparse, json, time
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
META_DIR   = WS / "test_images/nuscenes/v1.0-mini"
SAMPLES_DIR= WS / "test_images/nuscenes/samples/CAM_FRONT"
SWEEPS_DIR = WS / "test_images/nuscenes/sweeps/CAM_FRONT"
CONSEC_DIR = WS / "test_images/nuimages/images"

LABEL_NAMES = {
    1: "car",        2: "truck",       3: "bus",
    4: "trailer",    5: "constr_veh",  6: "pedestrian",
    7: "motorcycle", 8: "bicycle",     9: "traffic_cone",
    10: "barrier",
}

# nuImages 类别名称 → 模型 label_id 映射
NUIMG_TO_LABEL = {}
_MAPPING = {
    "vehicle.car":               1,
    "vehicle.truck":             2,
    "vehicle.bus.rigid":         3,
    "vehicle.bus.bendy":         3,
    "vehicle.trailer":           4,
    "vehicle.construction":      5,
    "human.pedestrian.adult":    6,
    "human.pedestrian.child":    6,
    "human.pedestrian.construction_worker": 6,
    "human.pedestrian.personal_mobility":   6,
    "human.pedestrian.police_officer":      6,
    "human.pedestrian.stroller":            6,
    "human.pedestrian.wheelchair":          6,
    "vehicle.motorcycle":        7,
    "vehicle.bicycle":           8,
    "movable_object.trafficcone":9,
    "movable_object.barrier":    10,
}
MAIN_LABELS = [1, 2, 3, 6, 7, 8]
CLS_COLOR = {
    1: (0, 200, 0), 2: (0, 100, 255), 3: (0, 0, 255),
    4: (255, 0, 200), 5: (128, 0, 128), 6: (255, 255, 0),
    7: (255, 128, 0), 8: (0, 255, 255), 9: (200, 200, 200), 10: (100, 200, 255),
}
ADAPTIVE_THR = {
    1: 0.50, 2: 0.50, 3: 0.40, 4: 0.50, 5: 0.50,
    6: 0.35, 7: 0.35, 8: 0.25, 9: 0.50, 10: 0.50,
}


# ─── TRT ──────────────────────────────────────────────────────────────────────
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
            bgr, 1.0/255.0, (self.W, self.H), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        _h2d(self._d_inp, blob)
        self._ctx.execute_async_v3(0)
        _cudart.cudaDeviceSynchronize()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n <= 0:
            return np.empty(0, np.float32), np.empty(0, np.int32), np.empty((0,4), np.float32)
        sc = np.empty(n, np.float32);      _d2h(sc, self._d_scores)
        lb = np.empty(n, np.int32);        _d2h(lb, self._d_labels)
        bx = np.empty(n*4, np.float32);    _d2h(bx, self._d_boxes)
        return sc, lb, bx.reshape(n,4)

    def close(self):
        _free(self._d_inp); _free(self._d_scores)
        _free(self._d_labels); _free(self._d_boxes)


def do_infer(inf, bgr, thr_fn):
    oh, ow = bgr.shape[:2]
    sx, sy = ow / inf.W, oh / inf.H
    sc_arr, lb_arr, bx_arr = inf.infer(bgr)
    dets = []
    for i in range(len(sc_arr)):
        sc = float(sc_arr[i]); lb = int(lb_arr[i])
        if not thr_fn(lb, sc): continue
        x1 = float(bx_arr[i,0]*sx); y1 = float(bx_arr[i,1]*sy)
        x2 = float(bx_arr[i,2]*sx); y2 = float(bx_arr[i,3]*sy)
        if x2 <= x1 or y2 <= y1: continue
        dets.append((lb, x1, y1, x2, y2, sc))
    return dets


def nms_merge(preds_list, iou_thr=0.5):
    def iou(a,b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua>0 else 0.0
    all_p=sorted([p for lst in preds_list for p in lst], key=lambda x:-x[5])
    used=[False]*len(all_p); kept=[]
    for i,p in enumerate(all_p):
        if used[i]: continue
        used[i]=True; kept.append(p)
        for j in range(i+1,len(all_p)):
            if used[j]: continue
            q=all_p[j]
            if p[0]==q[0] and iou(p[1:5],q[1:5])>=iou_thr: used[j]=True
    return kept


# ─── 加载 GT 标注 ─────────────────────────────────────────────────────────────
def load_gt():
    """返回 {filename_stem: [(label_id, x1, y1, x2, y2), ...]}"""
    cats_raw  = json.loads((META_DIR/"category.json").read_text())
    cat_name  = {c["token"]: c["name"] for c in cats_raw}
    # 建立 category_token → label_id 映射
    cat2label = {t: _MAPPING.get(n) for t,n in cat_name.items() if _MAPPING.get(n)}

    sd_raw  = json.loads((META_DIR/"sample_data.json").read_text())
    # 只取 CAM_FRONT keyframe
    sd_map  = {s["token"]: s["filename"]
               for s in sd_raw
               if "CAM_FRONT" in s.get("filename","")
               and "sweeps" not in s.get("filename","")
               and s.get("is_key_frame", False)}

    ann_raw = json.loads((META_DIR/"object_ann.json").read_text())
    gt = {}
    for a in ann_raw:
        if a["sample_data_token"] not in sd_map: continue
        lb = cat2label.get(a["category_token"])
        if lb is None: continue
        bbox = a.get("bbox")
        if not bbox or len(bbox) < 4: continue
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if x2 <= x1 or y2 <= y1: continue
        fname = Path(sd_map[a["sample_data_token"]]).name
        stem = Path(fname).stem
        gt.setdefault(stem, []).append((lb, x1, y1, x2, y2))
    return gt


def iou_box(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0


# ─── Part1: GT 精确评估 ───────────────────────────────────────────────────────
def eval_gt(name, images_with_gt, gt_map, run_fn, iou_thr=0.5):
    """
    对有GT的keyframe图片做 Recall/Precision.
    返回 {label_id: (tp, fp, fn)}
    """
    tp_all = {c:0 for c in range(1,11)}
    fp_all = {c:0 for c in range(1,11)}
    fn_all = {c:0 for c in range(1,11)}
    latencies = []

    for img_path in images_with_gt:
        stem = img_path.stem
        bgr  = cv2.imread(str(img_path))
        if bgr is None: continue

        t0 = time.perf_counter()
        dets = run_fn(bgr)
        latencies.append((time.perf_counter()-t0)*1000)

        gts = gt_map.get(stem, [])  # [(lb, x1,y1,x2,y2), ...]
        # 按类别分组
        gt_by_cls  = {}
        for lb,x1,y1,x2,y2 in gts:
            gt_by_cls.setdefault(lb,[]).append((x1,y1,x2,y2))

        det_by_cls = {}
        for lb,x1,y1,x2,y2,sc in sorted(dets, key=lambda d:-d[5]):
            det_by_cls.setdefault(lb,[]).append((x1,y1,x2,y2,sc))

        for c in range(1,11):
            c_gt  = gt_by_cls.get(c,[])
            c_det = det_by_cls.get(c,[])
            matched_gt = [False]*len(c_gt)
            tp_img = 0
            for dx1,dy1,dx2,dy2,_ in c_det:
                best_iou, best_j = 0.0, -1
                for j,(gx1,gy1,gx2,gy2) in enumerate(c_gt):
                    if matched_gt[j]: continue
                    iv = iou_box((dx1,dy1,dx2,dy2),(gx1,gy1,gx2,gy2))
                    if iv > best_iou: best_iou,best_j = iv,j
                if best_iou >= iou_thr and best_j >= 0:
                    matched_gt[best_j] = True
                    tp_img += 1
                    tp_all[c] += 1
                else:
                    fp_all[c] += 1
            fn_all[c] += len(c_gt) - tp_img

    return tp_all, fp_all, fn_all, latencies


# ─── Part2: 检测覆盖统计 ──────────────────────────────────────────────────────
def stat_coverage(images, run_fn):
    total_dets = {c:0 for c in range(1,11)}
    all_scores = {c:[] for c in range(1,11)}
    latencies  = []
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None: continue
        t0 = time.perf_counter()
        dets = run_fn(bgr)
        latencies.append((time.perf_counter()-t0)*1000)
        for lb,x1,y1,x2,y2,sc in dets:
            if lb in total_dets:
                total_dets[lb] += 1
                all_scores[lb].append(sc)
    return total_dets, all_scores, latencies


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()
    THR = args.thr

    # ─── 加载 GT ─────────────────────────────────────────────────────────────
    print("加载 GT 标注 ...")
    gt_map = load_gt()
    print(f"  CAM_FRONT keyframe 图片数(有GT): {len(gt_map)}")
    total_gt_boxes = sum(len(v) for v in gt_map.values())
    print(f"  GT 框总数: {total_gt_boxes}")

    # keyframe 图片路径
    kf_images = sorted(SAMPLES_DIR.glob("*.jpg"))
    kf_with_gt = [p for p in kf_images if p.stem in gt_map]
    print(f"  在 {SAMPLES_DIR.name} 中找到匹配图片: {len(kf_with_gt)}")

    # ─── 覆盖统计图片 ─────────────────────────────────────────────────────────
    sweep_imgs  = sorted(SWEEPS_DIR.glob("*.jpg"))[:200]
    consec_imgs = sorted(CONSEC_DIR.glob("*.jpg"))[:500] if CONSEC_DIR.exists() else []
    cov_images  = list(kf_images) + sweep_imgs + consec_imgs
    # 去重
    seen = set(); cov_images_dedup = []
    for p in cov_images:
        if p.stem not in seen: seen.add(p.stem); cov_images_dedup.append(p)
    cov_images = cov_images_dedup

    print(f"\n覆盖统计图片组成:")
    print(f"  keyframe samples : {len(kf_images)} 张 (8 个不同场景)")
    print(f"  mini sweeps      : {len(sweep_imgs)} 张 (8 个不同 log)")
    print(f"  连续 sweeps      : {len(consec_imgs)} 张 (同一 log, n003-2018-01-02)")
    print(f"  合计去重         : {len(cov_images)} 张")

    # ─── 加载引擎 ─────────────────────────────────────────────────────────────
    INPUT_W = 1242
    def load_inf(fname, h):
        p = MODELS_DIR/fname
        if not p.exists(): print(f"  [SKIP] {fname}"); return None
        sz = p.stat().st_size//1024**2
        print(f"  ✓ {fname}  ({sz}MB)  输入={h}×{INPUT_W}")
        return Inferencer(str(p), h, INPUT_W)

    print("\n加载 TensorRT 引擎 ...")
    inf500  = load_inf("faster_rcnn_500.engine",      500)
    inf375  = load_inf("faster_rcnn_375.engine",      375)
    inf500i = load_inf("faster_rcnn_500_int8.engine", 500)
    inf700  = load_inf("faster_rcnn_700.engine",      700)
    print()

    # 热身
    sample_bgr = cv2.imread(str(kf_images[0]))
    print("引擎热身 (5次) ...")
    for _ in range(5):
        for inf in [inf500,inf375,inf500i,inf700]:
            if inf: inf.infer(sample_bgr)
    print("完成\n")

    # ─── 配置表 ───────────────────────────────────────────────────────────────
    thr_fixed    = lambda lb,sc: sc >= THR
    thr_adaptive = lambda lb,sc: sc >= ADAPTIVE_THR.get(lb, THR)

    CONFIGS = []
    if inf500:
        _i=inf500
        CONFIGS.append(("基线  FP16_500h  thr=0.5",
                         lambda bgr,i=_i: do_infer(i,bgr,thr_fixed)))
    if inf375 and inf500:
        _a,_b=inf375,inf500
        CONFIGS.append(("方案1 多尺度(375+500) thr=0.5 [↑召回]",
                         lambda bgr,a=_a,b=_b: nms_merge([do_infer(a,bgr,thr_fixed),
                                                            do_infer(b,bgr,thr_fixed)])))
    if inf500i:
        _i=inf500i
        CONFIGS.append(("方案2 INT8_500h 自适应阈值   [↑速度]",
                         lambda bgr,i=_i: do_infer(i,bgr,thr_adaptive)))
    if inf375:
        _i=inf375
        CONFIGS.append(("方案3 FP16_375h  thr=0.5    [速度优先]",
                         lambda bgr,i=_i: do_infer(i,bgr,thr_fixed)))
    if inf700:
        _i=inf700
        CONFIGS.append(("附加  FP16_700h  thr=0.5    [训练分辨率]",
                         lambda bgr,i=_i: do_infer(i,bgr,thr_fixed)))

    # ─── 运行评估 ─────────────────────────────────────────────────────────────
    SEP = "═"*80
    results = []
    for cname, run_fn in CONFIGS:
        print(f"\n{'─'*80}")
        print(f"► 配置: {cname}")

        # Part1: GT 精确评估
        if kf_with_gt:
            tp, fp, fn, lats = eval_gt(cname, kf_with_gt, gt_map, run_fn, args.iou_thr)
            lat_mean = float(np.mean(lats)) if lats else 0.0
            print(f"\n  [Part 1] GT 评估  ({len(kf_with_gt)} 张 keyframe | IoU≥{args.iou_thr})")
            print(f"  {'类别':14s}{'GT框':>7}{'TP':>6}{'FP':>6}{'FN':>6}"
                  f"{'Recall':>9}{'Prec':>9}{'F1':>8}")
            print("  " + "─"*65)
            for c in MAIN_LABELS + [9, 10]:
                cname2 = LABEL_NAMES[c]
                gtc  = tp[c]+fn[c]
                rec  = tp[c]/(tp[c]+fn[c]) if (tp[c]+fn[c])>0 else 0.0
                prec = tp[c]/(tp[c]+fp[c]) if (tp[c]+fp[c])>0 else 0.0
                f1   = 2*rec*prec/(rec+prec) if (rec+prec)>0 else 0.0
                print(f"  {cname2:14s}{gtc:7d}{tp[c]:6d}{fp[c]:6d}{fn[c]:6d}"
                      f"{rec:9.1%}{prec:9.1%}{f1:8.1%}")
            total_tp = sum(tp[c] for c in MAIN_LABELS)
            total_gt = sum((tp[c]+fn[c]) for c in MAIN_LABELS)
            total_fp = sum(fp[c] for c in MAIN_LABELS)
            ov_rec  = total_tp/total_gt if total_gt>0 else 0.0
            ov_prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0
            ov_f1   = 2*ov_rec*ov_prec/(ov_rec+ov_prec) if (ov_rec+ov_prec)>0 else 0.0
            print("  " + "─"*65)
            print(f"  {'综合(主类)':14s}{total_gt:7d}{total_tp:6d}{total_fp:6d}"
                  f"{total_gt-total_tp:6d}"
                  f"{ov_rec:9.1%}{ov_prec:9.1%}{ov_f1:8.1%}")
            print(f"  延迟(keyframe): mean={lat_mean:.1f}ms")
        else:
            tp=fp=fn={c:0 for c in range(1,11)}; lat_mean=0; lats=[]
            ov_rec=ov_prec=ov_f1=0.0

        # Part2: 覆盖统计
        print(f"\n  [Part 2] 检测覆盖统计  ({len(cov_images)} 张)")
        td, sc_stat, lats2 = stat_coverage(cov_images, run_fn)
        lat2 = float(np.mean(lats2)) if lats2 else 0.0
        n = len(cov_images)
        print(f"  {'类别':14s}{'总检测':>8}{'每帧':>7}{'P50分':>8}{'P95分':>8}")
        print("  " + "─"*50)
        for c in MAIN_LABELS:
            cname2 = LABEL_NAMES[c]
            cnt = td[c]
            avg = cnt/n if n else 0.0
            scores = sc_stat[c]
            p50 = float(np.percentile(scores,50)) if scores else 0.0
            p95 = float(np.percentile(scores,95)) if scores else 0.0
            print(f"  {cname2:14s}{cnt:8d}{avg:7.2f}{p50:8.3f}{p95:8.3f}")
        all_cnt = sum(td[c] for c in MAIN_LABELS)
        print(f"  延迟(全部): mean={lat2:.1f}ms  每帧总检测: {all_cnt/n:.2f}")

        results.append({
            "name": cname,
            "tp": tp, "fp": fp, "fn": fn,
            "recall": ov_rec, "prec": ov_prec, "f1": ov_f1,
            "lat": lat_mean,
            "td": td, "lat2": lat2,
        })

    # ─── 综合对比表 ───────────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print(f"  ★  综合对比表")
    print(SEP)

    # GT 评估对比
    if kf_with_gt:
        print(f"\n  [GT 精确评估] {len(kf_with_gt)} 张 keyframe, IoU≥{args.iou_thr}, thr={THR}")
        print(f"  {'配置':45s}{'Recall':>9}{'Prec':>9}{'F1':>8}{'延迟':>8}")
        print("  " + "─"*80)
        for r in results:
            print(f"  {r['name']:45s}{r['recall']:9.1%}{r['prec']:9.1%}"
                  f"{r['f1']:8.1%}{r['lat']:8.1f}ms")

        # 按类别对比
        print(f"\n  逐类 Recall (GT评估, 主要类别):")
        print(f"  {'类别':14s}" + "".join(f" {r['name'][:18]:>18s}" for r in results))
        print("  " + "─"*90)
        for c in MAIN_LABELS:
            cname2 = LABEL_NAMES[c]
            row = f"  {cname2:14s}"
            for r in results:
                rec = r['tp'][c]/(r['tp'][c]+r['fn'][c]) if (r['tp'][c]+r['fn'][c])>0 else 0.0
                row += f" {rec:17.1%} "
            print(row)

    # 检测统计对比
    n = len(cov_images)
    print(f"\n  [覆盖统计] {n} 张图片, thr={THR}")
    print(f"  {'配置':45s}{'每帧总检测':>11}{'延迟ms':>9}")
    print("  " + "─"*70)
    for r in results:
        all_cnt = sum(r['td'][c] for c in MAIN_LABELS)
        print(f"  {r['name']:45s}{all_cnt/n:11.2f}{r['lat2']:9.1f}ms")

    print(f"\n{SEP}")
    print("  注: GT评估仅使用8张有标注的keyframe图片(样本较少,供参考)")
    print("      覆盖统计使用无标注sweeps,反映真实场景检测能力")

    # ─── 保存可视化 ───────────────────────────────────────────────────────────
    if kf_with_gt and results:
        ref_run = results[0]  # 用第一个配置(基线)可视
        run_fn0 = CONFIGS[0][1]
        vis_dir = WS/"test_images/compare_results/nuimages_gt_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n保存 keyframe 可视化 → {vis_dir}")
        for img_p in kf_with_gt:
            bgr = cv2.imread(str(img_p))
            if bgr is None: continue
            dets = run_fn0(bgr)
            gt_boxes = gt_map.get(img_p.stem, [])
            # 绘制GT（绿）
            for lb,x1,y1,x2,y2 in gt_boxes:
                cv2.rectangle(bgr,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(bgr, f"GT:{LABEL_NAMES.get(lb,'?')}",
                           (int(x1),max(int(y1)-14,12)),
                           cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1,cv2.LINE_AA)
            # 绘制检测（红）
            for lb,x1,y1,x2,y2,sc in dets:
                cv2.rectangle(bgr,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                cv2.putText(bgr, f"{LABEL_NAMES.get(lb,'?')}:{sc:.2f}",
                           (int(x1),max(int(y2)+14,12)),
                           cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1,cv2.LINE_AA)
            out = vis_dir/f"{img_p.stem}_gt_det.jpg"
            h,w = bgr.shape[:2]
            bgr_s = cv2.resize(bgr,(int(w*0.5),int(h*0.5)))
            cv2.imwrite(str(out), bgr_s)
        print(f"  绿框=GT  红框=检测预测  共 {len(kf_with_gt)} 张")

    # 关闭引擎
    for inf in [inf500, inf375, inf500i, inf700]:
        if inf: inf.close()


if __name__ == "__main__":
    main()
