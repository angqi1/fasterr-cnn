#!/usr/bin/env python3
"""
eval_nuscenes.py — 使用 nuScenes mini 数据集评估 TensorRT 推理引擎

与 KITTI 评估的核心区别:
  ✓ 图像分辨率匹配: NuScenes 1600×900 vs KITTI 1224×370
  ✓ 无需类别映射: 模型输出与 NuScenes 标注直接对照（同为10类）
  ✓ 全类别评估: bicycle/bus/pedestrian 均有充足样本
  ✓ 反映真实性能: 模型在 NuScenes 上训练，NuScenes 上评估才准确

测试配置 (均以 thr=0.5 为基准，对比3种优化方案):
  基线:  FP16_500h  thr=0.5
  方案1: 多尺度融合  FP16_375h + FP16_500h  thr=0.5
  方案2: INT8_500h  自适应阈值 (bus=0.40 / ped=0.35 / bicycle=0.25)
  方案3: FP16_375h  thr=0.5  (速度优先)
  附加:  FP16_700h  thr=0.5  (训练分辨率匹配，理论最优)

运行前提: 已下载 nuScenes mini -> ~/ros2_ws/test_images/nuscenes/
  如未下载: python3 download_nuscenes_mini.py --instructions

用法:
  python3 eval_nuscenes.py
  python3 eval_nuscenes.py --max-images 200 --cameras CAM_FRONT
  python3 eval_nuscenes.py --dataroot /your/nuscenes/path
"""

import argparse, json, sys, time
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
IOU_THR    = 0.5

# ─── nuScenes 类别 → 模型 label_id ────────────────────────────────────────────
# 完整映射（与训练时一致）
CAT_TO_LABEL = {
    "vehicle.car":                           1,
    "vehicle.truck":                         2,
    "vehicle.bus.bendy":                     3,
    "vehicle.bus.rigid":                     3,
    "vehicle.trailer":                       4,
    "vehicle.construction":                  5,
    "human.pedestrian.adult":                6,
    "human.pedestrian.child":                6,
    "human.pedestrian.wheelchair":           6,
    "human.pedestrian.stroller":             6,
    "human.pedestrian.personal_mobility":    6,
    "human.pedestrian.police_officer":       6,
    "human.pedestrian.construction_worker":  6,
    "vehicle.motorcycle":                    7,
    "vehicle.bicycle":                       8,
    "movable_object.trafficcone":            9,
    "movable_object.barrier":               10,
    # 以下在评估中忽略
    "movable_object.debris":                None,
    "movable_object.pushable_pullable":     None,
    "static_object.bicycle_rack":           None,
    "animal":                               None,
}

# label_id → 中文名（用于打印）
LABEL_NAMES = {
    1: "car", 2: "truck", 3: "bus", 4: "trailer",
    5: "constr_veh", 6: "pedestrian", 7: "motorcycle",
    8: "bicycle", 9: "traffic_cone", 10: "barrier",
}

# 评估的关注类别（与模型主要输出对应）
EVAL_LABELS = [1, 2, 3, 6, 7, 8]  # car truck bus pedestrian motorcycle bicycle

# 方案2 自适应阈值
ADAPTIVE_THR = {
    1: 0.50, 2: 0.50, 3: 0.40, 4: 0.50, 5: 0.50,
    6: 0.35, 7: 0.35, 8: 0.25, 9: 0.50, 10: 0.50,
}

# ─── TRT 推理器 ────────────────────────────────────────────────────────────────
class Inferencer:
    def __init__(self, engine_path, input_h, input_w=1600):
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

    def infer_raw(self, bgr):
        """返回原始推理结果（坐标为模型输入尺寸下）"""
        blob = cv2.dnn.blobFromImage(
            bgr, 1.0 / 255.0, (self.W, self.H), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        _h2d(self._d_inp, blob)
        self._ctx.execute_async_v3(0)
        _cudart.cudaDeviceSynchronize()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n <= 0:
            return np.empty(0), np.empty(0, np.int32), np.empty((0, 4))
        sc = np.empty(n, np.float32);      _d2h(sc, self._d_scores)
        lb = np.empty(n, np.int32);        _d2h(lb, self._d_labels)
        bx = np.empty(n * 4, np.float32);  _d2h(bx, self._d_boxes)
        return sc, lb, bx.reshape(n, 4)

    def close(self):
        _free(self._d_inp); _free(self._d_scores)
        _free(self._d_labels); _free(self._d_boxes)


def do_infer(inf, bgr, thr_fn):
    """推理一张图，坐标还原到原图，过滤低分，返回 [(cls, x1,y1,x2,y2, score)]"""
    oh, ow = bgr.shape[:2]
    sx, sy = ow / inf.W, oh / inf.H
    sc_arr, lb_arr, bx_arr = inf.infer_raw(bgr)
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


def nms_merge(preds_list, iou_thr=0.5):
    """多尺度预测融合 NMS"""
    all_preds = sorted([p for lst in preds_list for p in lst], key=lambda x: -x[5])
    used = [False] * len(all_preds)
    kept = []
    for i, p in enumerate(all_preds):
        if used[i]: continue
        used[i] = True; kept.append(p)
        for j in range(i + 1, len(all_preds)):
            if used[j]: continue
            q = all_preds[j]
            if p[0] == q[0] and iou_2d(p[1:5], q[1:5]) >= iou_thr:
                used[j] = True
    return kept


# ─── IoU ─────────────────────────────────────────────────────────────────────
def iou_2d(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ─── nuScenes GT 加载（3D bbox 投影到 2D）─────────────────────────────────────
def load_nuscenes_gt(nusc, samples, cameras):
    """
    加载样本列表中所有摄像头的 2D GT 框。
    返回: {img_token: [(label_id, x1, y1, x2, y2), ...]}
    """
    from nuscenes.utils.geometry_utils import view_points, BoxVisibility
    from pyquaternion import Quaternion

    gt_dict = {}
    for sample in samples:
        for cam in cameras:
            if cam not in sample["data"]:
                continue
            sd_token = sample["data"][cam]
            img_path, boxes, cam_intrinsic = nusc.get_sample_data(
                sd_token, box_vis_level=BoxVisibility.ANY)
            if not Path(img_path).exists():
                continue

            gt_boxes = []
            for box in boxes:
                # 获取类别名（前缀到最细粒度）
                ann = nusc.get("sample_annotation", box.token)
                cat_name = ann["category_name"]
                label = CAT_TO_LABEL.get(cat_name)
                if label is None:
                    continue

                # 3D 角点投影到 2D
                corners_3d = box.corners()          # (3, 8)
                if corners_3d[2, :].min() < 0.1:   # 确保在相机前方
                    continue
                pts = view_points(corners_3d, np.array(cam_intrinsic), normalize=True)
                x1, y1 = float(pts[0].min()), float(pts[1].min())
                x2, y2 = float(pts[0].max()), float(pts[1].max())
                # 剪裁到图像边界（NuScenes 图像 1600×900）
                x1 = max(0.0, x1); y1 = max(0.0, y1)
                x2 = min(1600.0, x2); y2 = min(900.0, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                gt_boxes.append((label, x1, y1, x2, y2))

            gt_dict[sd_token] = (img_path, gt_boxes)

    return gt_dict


# ─── GT 匹配 ──────────────────────────────────────────────────────────────────
def match_gt(gt_list, pred_list):
    matched_gt = set(); matched_pred = set()
    for pi, pred in sorted(enumerate(pred_list), key=lambda x: -x[1][5]):
        pb = pred[1:5]; pcls = pred[0]
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_list):
            if gi in matched_gt: continue
            if gt[0] != pcls: continue
            v = iou_2d(pb, gt[1:5])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= IOU_THR and best_gi >= 0:
            matched_gt.add(best_gi); matched_pred.add(pi)
    return matched_gt, matched_pred


# ─── 指标计算 ─────────────────────────────────────────────────────────────────
def compute_metrics(gt_dict, per_image):
    tot_gt = tot_pred = tot_hit = 0
    cls_gt   = {c: 0 for c in EVAL_LABELS}
    cls_pred = {c: 0 for c in EVAL_LABELS}
    cls_hit  = {c: 0 for c in EVAL_LABELS}

    for token, (img_path, gt_list) in gt_dict.items():
        pred_list = per_image.get(token, [])
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
            if c in cls_pred:
                cls_pred[c] += 1

    def safe(h, d): return h / d * 100 if d > 0 else 0.0
    overall = (tot_gt, tot_pred, tot_hit, tot_gt - tot_hit, tot_pred - tot_hit,
               safe(tot_hit, tot_gt), safe(tot_hit, tot_pred))
    per_cls = {}
    for c in EVAL_LABELS:
        g = cls_gt[c]; ph = cls_hit[c]; p = cls_pred[c]
        per_cls[c] = (g, p, ph, g - ph, max(0, p - ph),
                      safe(ph, g), safe(ph, p))
    return overall, per_cls


# ─── 打印 ─────────────────────────────────────────────────────────────────────
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
    print(f"     {'类别':14s}{'GT':>5}{'Pred':>7}{'Hit':>6}{'Miss':>6}{'FP':>6}"
          f"{'Recall':>9}{'Prec':>9}")
    print(f"     {'─'*72}")
    for c in EVAL_LABELS:
        cname = LABEL_NAMES.get(c, str(c))
        cg, cp, ch, cm, cfp, cr, cpr = per_cls[c]
        bar = "▇" * int(cr / 5)
        print(f"     {cname:14s}{cg:5d}{cp:7d}{ch:6d}{cm:6d}{cfp:6d}"
              f"{cr:8.1f}%{cpr:8.1f}%  {bar}")
    return lat_mean, lat_p95, rec, prec, f1


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot",   default="",
                    help="nuScenes 数据根目录（含 v1.0-mini/ 和 samples/）")
    ap.add_argument("--version",    default="v1.0-mini")
    ap.add_argument("--cameras",    default="CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT",
                    help="使用的摄像头列表，逗号分隔")
    ap.add_argument("--max-images", type=int, default=500,
                    help="最多使用图像数（默认500）")
    args = ap.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")]

    # ── 查找数据目录 ────────────────────────────────────────────────────────────
    dataroot = None
    if args.dataroot:
        cand = Path(args.dataroot)
        if (cand / "v1.0-mini" / "scene.json").exists():
            dataroot = str(cand)
    if dataroot is None:
        for cand in [WS / "test_images/nuscenes",
                     Path("/data/nuscenes"),
                     Path("/home/nvidia/nuscenes")]:
            if (cand / "v1.0-mini" / "scene.json").exists():
                dataroot = str(cand)
                break

    if dataroot is None:
        print("\n[ERROR] 未找到 nuScenes mini 数据集！")
        print("请先运行: python3 download_nuscenes_mini.py --instructions")
        sys.exit(1)

    print(f"[数据集] nuScenes {args.version}  dataroot={dataroot}")

    # ── 加载 nuScenes devkit ────────────────────────────────────────────────────
    try:
        from nuscenes import NuScenes
    except ImportError:
        print("[ERROR] 未安装 nuscenes-devkit: pip3 install nuscenes-devkit")
        sys.exit(1)

    print("初始化 nuScenes devkit ...")
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=False)

    # 取 mini 的所有 scene 下的 sample（关键帧）
    all_samples = nusc.sample
    print(f"  场景数: {len(nusc.scene)}  样本总数: {len(all_samples)}")

    # ── 加载 GT（3D→2D 投影）──────────────────────────────────────────────────
    print(f"投影 GT 3D→2D（摄像头: {cameras}）...")
    gt_dict = load_nuscenes_gt(nusc, all_samples, cameras)

    # 限制到 max_images 条
    tokens = list(gt_dict.keys())[:args.max_images]
    gt_dict = {t: gt_dict[t] for t in tokens}

    total_gt = sum(len(v[1]) for v in gt_dict.values())
    print(f"  有效图像数: {len(gt_dict)}  GT 总目标数: {total_gt}")
    if total_gt == 0:
        print("[ERROR] GT 目标数为0，请检查数据集路径和摄像头配置")
        sys.exit(1)

    # GT 类别分布
    cls_cnt = {c: 0 for c in EVAL_LABELS}
    for _, (_, gt_boxes) in gt_dict.items():
        for gt in gt_boxes:
            c = gt[0]
            if c in cls_cnt:
                cls_cnt[c] += 1
    print("  GT 类别分布:")
    for c in EVAL_LABELS:
        pct = cls_cnt[c] / total_gt * 100 if total_gt > 0 else 0
        print(f"    {LABEL_NAMES[c]:14s}: {cls_cnt[c]:5d} ({pct:.1f}%)")
    print()

    # ── 加载引擎 ───────────────────────────────────────────────────────────────
    # NuScenes 图像宽度 = 1600，高度 = 900
    # 引擎输入宽度固定用 1600（覆盖 NuScenes 完整宽度）
    INPUT_W = 1600
    def load_inf(fname, h):
        p = MODELS_DIR / fname
        if not p.exists():
            print(f"  [SKIP] {fname} 不存在")
            return None
        sz = p.stat().st_size // 1024 ** 2
        print(f"  ✓ 加载 {fname}  ({sz} MB)  输入={h}×{INPUT_W}")
        return Inferencer(str(p), h, INPUT_W)

    print("加载 TensorRT 引擎 ...")
    inf375  = load_inf("faster_rcnn_375.engine", 375)
    inf500  = load_inf("faster_rcnn_500.engine", 500)
    inf500i = load_inf("faster_rcnn_500_int8.engine", 500)
    inf700  = load_inf("faster_rcnn_700.engine", 700)
    print()

    # ── 阈值函数 ───────────────────────────────────────────────────────────────
    thr_fixed    = lambda lb, sc: sc >= 0.50
    thr_adaptive = lambda lb, sc: sc >= ADAPTIVE_THR.get(lb, 0.50)

    # ── 热身 ──────────────────────────────────────────────────────────────────
    first_img = cv2.imread(list(gt_dict.values())[0][0])
    print("引擎热身中 ...")
    for _ in range(5):
        for inf in [inf375, inf500, inf500i, inf700]:
            if inf: inf.infer_raw(first_img)
    print("热身完成\n")

    # ── 配置表 ────────────────────────────────────────────────────────────────
    CONFIGS = []
    if inf500:
        _i = inf500
        CONFIGS.append({
            "name": "基线:  FP16_500h  thr=0.5",
            "run":  lambda bgr, i=_i: do_infer(i, bgr, thr_fixed),
        })
    if inf375 and inf500:
        _a, _b = inf375, inf500
        CONFIGS.append({
            "name": "方案1: 多尺度融合 (375h+500h) thr=0.5  [↑召回]",
            "run":  lambda bgr, a=_a, b=_b: nms_merge([
                        do_infer(a, bgr, thr_fixed),
                        do_infer(b, bgr, thr_fixed),
                    ]),
        })
    if inf500i:
        _i = inf500i
        CONFIGS.append({
            "name": "方案2: INT8_500h  自适应阈值           [↑速度+召回兼顾]",
            "run":  lambda bgr, i=_i: do_infer(i, bgr, thr_adaptive),
        })
    if inf375:
        _i = inf375
        CONFIGS.append({
            "name": "方案3: FP16_375h  thr=0.5              [↑速度优先]",
            "run":  lambda bgr, i=_i: do_infer(i, bgr, thr_fixed),
        })
    if inf700:
        _i = inf700
        CONFIGS.append({
            "name": "附加:  FP16_700h  thr=0.5              [匹配NuScenes分辨率]",
            "run":  lambda bgr, i=_i: do_infer(i, bgr, thr_fixed),
        })

    # ── 逐配置推理评估 ────────────────────────────────────────────────────────
    summary = []
    for cfg in CONFIGS:
        per_image = {}
        latencies = []
        for token, (img_path, _) in gt_dict.items():
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            t0   = time.perf_counter()
            dets = cfg["run"](bgr)
            latencies.append((time.perf_counter() - t0) * 1000)
            per_image[token] = dets

        overall, per_cls = compute_metrics(gt_dict, per_image)
        lm, lp, rec, prec, f1 = print_result(cfg["name"], overall, per_cls, latencies)
        summary.append((cfg["name"], rec, prec, f1, lm, lp, per_cls, overall))

    # ── 综合对比表 ────────────────────────────────────────────────────────────
    W = 110
    print(f"\n{'═'*W}")
    print(f"  ★ 综合对比  ({len(gt_dict)} 张 nuScenes 图 | IoU≥{IOU_THR} | 基线 thr=0.5)")
    print(f"{'═'*W}")
    print(f"  {'配置':48s}{'GT':>6}{'Pred':>7}{'Hit':>6}{'FP':>7}"
          f"{'Recall':>9}{'Prec':>8}{'F1':>7}{'延迟ms':>8}{'P95ms':>8}")
    print("  " + "─" * (W - 2))
    for name, rec, prec, f1, lat, p95, _, overall in summary:
        gt, pred, hit, *_, = overall
        fp = overall[4]
        print(f"  {name:48s}{gt:6d}{pred:7d}{hit:6d}{fp:7d}"
              f"{rec:8.1f}%{prec:7.1f}%{f1:6.1f}%{lat:8.1f}{p95:8.1f}")
    print("  " + "─" * (W - 2))

    # ── 逐类 Recall 对比 ──────────────────────────────────────────────────────
    print(f"\n  逐类 Recall（%）")
    cls_hdr = f"  {'配置':48s}" + "".join(
        f" {LABEL_NAMES[c]:>11s}" for c in EVAL_LABELS)
    print(cls_hdr)
    print("  " + "─" * (W - 2))
    for name, rec, prec, f1, lat, p95, per_cls, _ in summary:
        row = f"  {name:48s}"
        for c in EVAL_LABELS:
            row += f" {per_cls[c][5]:10.1f}%"
        print(row)
    print("  " + "─" * (W - 2))

    # ── 逐类 Precision 对比 ───────────────────────────────────────────────────
    print(f"\n  逐类 Precision（%）")
    print(cls_hdr)
    print("  " + "─" * (W - 2))
    for name, rec, prec, f1, lat, p95, per_cls, _ in summary:
        row = f"  {name:48s}"
        for c in EVAL_LABELS:
            row += f" {per_cls[c][6]:10.1f}%"
        print(row)
    print("  " + "─" * (W - 2))

    # ── 最优方案推荐 ──────────────────────────────────────────────────────────
    if summary:
        best_rec  = max(summary, key=lambda x: x[1])
        best_prec = max(summary, key=lambda x: x[2])
        best_f1   = max(summary, key=lambda x: x[3])
        best_spd  = min(summary, key=lambda x: x[4])
        print(f"\n  ★ 最高 Recall   : {best_rec[0].strip()}  →  {best_rec[1]:.1f}%")
        print(f"  ★ 最高 Precision: {best_prec[0].strip()}  →  {best_prec[2]:.1f}%")
        print(f"  ★ 最高 F1       : {best_f1[0].strip()}  →  {best_f1[3]:.1f}%")
        print(f"  ★ 最快推理      : {best_spd[0].strip()}  →  {best_spd[4]:.1f}ms")

    # ── 关闭引擎 ──────────────────────────────────────────────────────────────
    for inf in [inf375, inf500, inf500i, inf700]:
        if inf: inf.close()

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out = WS / "test_images/compare_results/eval_nuscenes_summary.json"
    out.parent.mkdir(exist_ok=True)
    save = [
        {
            "config":              n,
            "recall":              r,
            "precision":           p,
            "f1":                  f,
            "lat_mean_ms":         lm,
            "lat_p95_ms":          lp,
            "per_class_recall":    {LABEL_NAMES[c]: pc[5] for c, pc in pcls.items()},
            "per_class_precision": {LABEL_NAMES[c]: pc[6] for c, pc in pcls.items()},
            "per_class_gt":        {LABEL_NAMES[c]: pc[0] for c, pc in pcls.items()},
        }
        for n, r, p, f, lm, lp, pcls, _ in summary
    ]
    json.dump(save, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
