#!/usr/bin/env python3
"""
eval_nuimages.py — 在 nuImages/nuScenes CAM_FRONT 图像上推理验证
                   （sweep 帧，分辨率 1600×900，与训练数据同域）

与 KITTI 评估的本质差异:
  ✓ 图像来自训练同源（nuScenes/nuImages，新加坡/波士顿街景）
  ✓ 分辨率 1600×900 — FP16_700h 内部映射 360×640 恰好是训练分辨率
  ✓ 视角为前视摄像头，与训练样本高度一致
  ✗ Sweep 帧无 2D GT 标注 → 故本脚本评估推理质量而非 Recall/Precision

评估内容:
  1. 各配置逐类平均检测数 / 每帧检测总数
  2. 各类别置信度分布（P50 / P95）
  3. 推理延迟（mean / P95）
  4. 保存 20 张可视化样本（带框图像）
  5. 对比表一览

测试配置（均以 thr=0.5 为 Precision 基准）:
  基线:  FP16_500h  thr=0.5
  方案1: 多尺度融合 FP16_375h + FP16_500h  thr=0.5
  方案2: INT8_500h  自适应阈值 (bus=0.40/ped=0.35/bicycle=0.25)
  方案3: FP16_375h  thr=0.5   (速度优先)
  附加:  FP16_700h  thr=0.5   (训练分辨率精确匹配)

用法:
  python3 eval_nuimages.py
  python3 eval_nuimages.py --img-dir /path/to/images --max-images 200
  python3 eval_nuimages.py --vis-count 30   # 保存30张可视化
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

# nuScenes 模型输出 label_id → 类别名
LABEL_NAMES = {
    1: "car",        2: "truck",       3: "bus",
    4: "trailer",    5: "constr_veh",  6: "pedestrian",
    7: "motorcycle", 8: "bicycle",     9: "traffic_cone",
    10: "barrier",
}
# 关注的主要类别（与训练数据分布匹配的前景目标）
MAIN_LABELS = [1, 2, 3, 6, 7, 8]

# 方案2 自适应阈值
ADAPTIVE_THR = {
    1: 0.50, 2: 0.50, 3: 0.40, 4: 0.50, 5: 0.50,
    6: 0.35, 7: 0.35, 8: 0.25, 9: 0.50, 10: 0.50,
}

# 可视化颜色（BGR）
CLS_COLOR = {
    1: (0, 200, 0),     # car        绿
    2: (0, 100, 255),   # truck      橙
    3: (0, 0, 255),     # bus        红
    4: (255, 0, 200),   # trailer    粉
    5: (128, 0, 128),   # constr_veh 紫
    6: (255, 255, 0),   # pedestrian 青
    7: (255, 128, 0),   # motorcycle 蓝
    8: (0, 255, 255),   # bicycle    黄
    9: (200, 200, 200), # cone       灰
    10: (100, 200, 255),# barrier    浅蓝
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
        sc = np.empty(n, np.float32);      _d2h(sc, self._d_scores)
        lb = np.empty(n, np.int32);        _d2h(lb, self._d_labels)
        bx = np.empty(n * 4, np.float32);  _d2h(bx, self._d_boxes)
        return sc, lb, bx.reshape(n, 4)

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
        x1 = float(bx_arr[i, 0] * sx); y1 = float(bx_arr[i, 1] * sy)
        x2 = float(bx_arr[i, 2] * sx); y2 = float(bx_arr[i, 3] * sy)
        if x2 <= x1 or y2 <= y1: continue
        dets.append((lb, x1, y1, x2, y2, sc))
    return dets


def nms_merge(preds_list, iou_thr=0.5):
    def iou(a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua>0 else 0.0
    all_p = sorted([p for lst in preds_list for p in lst], key=lambda x:-x[5])
    used = [False]*len(all_p); kept=[]
    for i,p in enumerate(all_p):
        if used[i]: continue
        used[i]=True; kept.append(p)
        for j in range(i+1,len(all_p)):
            if used[j]: continue
            q=all_p[j]
            if p[0]==q[0] and iou(p[1:5],q[1:5])>=iou_thr:
                used[j]=True
    return kept


# ─── 检测统计 ─────────────────────────────────────────────────────────────────
def compute_stats(per_image):
    """
    返回:
      total_dets:  {label: total_count}
      all_scores:  {label: [scores...]}
    """
    total_dets = {c: 0 for c in range(1, 11)}
    all_scores = {c: [] for c in range(1, 11)}
    for dets in per_image.values():
        for lb, x1, y1, x2, y2, sc in dets:
            if lb in total_dets:
                total_dets[lb] += 1
                all_scores[lb].append(sc)
    return total_dets, all_scores


# ─── 可视化绘图 ───────────────────────────────────────────────────────────────
def draw_dets(bgr, dets, scale=0.5):
    """在图像上绘制检测框，返回缩放后图像"""
    img = bgr.copy()
    for lb, x1, y1, x2, y2, sc in dets:
        color = CLS_COLOR.get(lb, (200, 200, 200))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label_str = f"{LABEL_NAMES.get(lb,'?')} {sc:.2f}"
        cv2.putText(img, label_str, (int(x1), max(int(y1)-4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# ─── 打印配置结果 ─────────────────────────────────────────────────────────────
def print_stats(name, n_imgs, total_dets, all_scores, latencies):
    lat_mean = float(np.mean(latencies))
    lat_p95  = float(np.percentile(latencies, 95))
    total_all = sum(total_dets.values())
    per_img   = total_all / n_imgs if n_imgs else 0
    SEP = "─" * 70
    print(f"\n  {SEP}")
    print(f"  ▶  {name}")
    print(f"     图像数: {n_imgs}  总检测数: {total_all}  每帧平均: {per_img:.1f}")
    print(f"     延迟: mean={lat_mean:.1f}ms  P95={lat_p95:.1f}ms")
    print(f"     {'类别':14s}{'总数':>7}{'每帧':>7}{'P50分':>8}{'P95分':>8}")
    print(f"     {'─'*46}")
    for c in MAIN_LABELS:
        cname = LABEL_NAMES[c]
        cnt   = total_dets[c]
        avg   = cnt / n_imgs if n_imgs else 0
        scores = all_scores[c]
        p50, p95 = (float(np.percentile(scores, 50)), float(np.percentile(scores, 95))) \
                   if scores else (0.0, 0.0)
        bar = "▇" * min(int(avg), 30)
        print(f"     {cname:14s}{cnt:7d}{avg:7.2f}{p50:8.3f}{p95:8.3f}  {bar}")
    return lat_mean, lat_p95, per_img


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir",    default="",   help="图像目录（默认自动查找）")
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--vis-count",  type=int, default=20, help="保存可视化图像数")
    args = ap.parse_args()

    # ── 查找图像目录 ────────────────────────────────────────────────────────────
    img_dir = None
    if args.img_dir:
        img_dir = Path(args.img_dir)
    else:
        for cand in [WS / "test_images/nuimages/images",
                     WS / "test_images/nuimages"]:
            imgs = list(cand.glob("*.jpg")) + list(cand.glob("*.png")) \
                   if cand.exists() else []
            if len(imgs) >= 5:
                img_dir = cand
                break

    if img_dir is None or not img_dir.exists():
        print("[ERROR] 未找到 nuImages 图像目录")
        print("请先运行: python3 download_nuimages.py")
        sys.exit(1)

    images = sorted(img_dir.glob("*.jpg"))[:args.max_images]
    images += sorted(img_dir.glob("*.png"))[:max(0, args.max_images - len(images))]
    images = images[:args.max_images]

    if not images:
        print(f"[ERROR] {img_dir} 中无图像文件")
        sys.exit(1)

    print(f"[数据集] nuImages CAM_FRONT sweep  图像数: {len(images)}")
    print(f"  路径: {img_dir}")
    # 检查第一张图分辨率
    sample = cv2.imread(str(images[0]))
    if sample is not None:
        print(f"  分辨率: {sample.shape[1]}×{sample.shape[0]}  (期望 1600×900)")
    print()

    # ── 引擎加载 ───────────────────────────────────────────────────────────────
    # 引擎编译时 width=1242（KITTI宽度），cv2.dnn.blobFromImage 自动缩放 1600→1242
    INPUT_W = 1242
    def load_inf(fname, h):
        p = MODELS_DIR / fname
        if not p.exists():
            print(f"  [SKIP] {fname} 不存在");  return None
        sz = p.stat().st_size // 1024**2
        print(f"  ✓ 加载 {fname}  ({sz} MB)  输入={h}×{INPUT_W}")
        return Inferencer(str(p), h, INPUT_W)

    print("加载 TensorRT 引擎 ...")
    inf375  = load_inf("faster_rcnn_375.engine", 375)
    inf500  = load_inf("faster_rcnn_500.engine", 500)
    inf500i = load_inf("faster_rcnn_500_int8.engine", 500)
    inf700  = load_inf("faster_rcnn_700.engine", 700)
    print()

    # ── 热身 ──────────────────────────────────────────────────────────────────
    print("引擎热身 ...")
    for _ in range(5):
        for inf in [inf375, inf500, inf500i, inf700]:
            if inf: inf.infer(sample)
    print("完成\n")

    # ── 配置表 ────────────────────────────────────────────────────────────────
    thr_fixed    = lambda lb, sc: sc >= 0.50
    thr_adaptive = lambda lb, sc: sc >= ADAPTIVE_THR.get(lb, 0.50)

    CONFIGS = []
    if inf500:
        _i=inf500
        CONFIGS.append({"name":"基线:  FP16_500h  thr=0.5",
                         "run": lambda bgr,i=_i: do_infer(i,bgr,thr_fixed)})
    if inf375 and inf500:
        _a,_b=inf375,inf500
        CONFIGS.append({"name":"方案1: 多尺度融合 (375h+500h) thr=0.5  [↑召回]",
                         "run": lambda bgr,a=_a,b=_b: nms_merge([do_infer(a,bgr,thr_fixed),
                                                                    do_infer(b,bgr,thr_fixed)])})
    if inf500i:
        _i=inf500i
        CONFIGS.append({"name":"方案2: INT8_500h  自适应阈值           [↑速度+召回]",
                         "run": lambda bgr,i=_i: do_infer(i,bgr,thr_adaptive)})
    if inf375:
        _i=inf375
        CONFIGS.append({"name":"方案3: FP16_375h  thr=0.5              [↑速度优先]",
                         "run": lambda bgr,i=_i: do_infer(i,bgr,thr_fixed)})
    if inf700:
        _i=inf700
        CONFIGS.append({"name":"附加:  FP16_700h  thr=0.5              [训练分辨率匹配]",
                         "run": lambda bgr,i=_i: do_infer(i,bgr,thr_fixed)})

    # 可视化输出目录
    vis_dir = WS / "test_images/compare_results/nuimages_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── 逐配置推理 ────────────────────────────────────────────────────────────
    summary = []
    for cfg in CONFIGS:
        per_image  = {}
        latencies  = []
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            t0   = time.perf_counter()
            dets = cfg["run"](bgr)
            latencies.append((time.perf_counter() - t0) * 1000)
            per_image[img_path.stem] = (bgr, dets)

        total_dets, all_scores = compute_stats(
            {k: v[1] for k, v in per_image.items()})
        lm, lp, avg_dets = print_stats(
            cfg["name"], len(per_image), total_dets, all_scores, latencies)
        summary.append((cfg["name"], lm, lp, avg_dets, total_dets, all_scores))

        # 保存可视化（仅第一个配置保存，避免磁盘占满）
        if summary and len(summary) == 1 and args.vis_count > 0:
            print(f"\n  保存 {args.vis_count} 张可视化 → {vis_dir}")
            items = list(per_image.items())[:args.vis_count]
            for stem, (bgr, dets) in items:
                out = vis_dir / f"{stem}_det.jpg"
                cv2.imwrite(str(out), draw_dets(bgr, dets, scale=0.5))
            print(f"  已保存 {len(items)} 张（缩放50%，长边约800px）")

    # ── 综合对比表 ────────────────────────────────────────────────────────────
    W = 100
    print(f"\n{'═'*W}")
    n_img = len(images)
    print(f"  ★ 综合对比  ({n_img} 张 nuImages CAM_FRONT 图 | 图像分辨率 1600×900)")
    print(f"{'═'*W}")
    print(f"  {'配置':48s}{'每帧检测':>9}{'延迟ms':>9}{'P95ms':>8}")
    print("  " + "─" * (W - 2))
    for name, lm, lp, avg, td, _ in summary:
        print(f"  {name:48s}{avg:9.2f}{lm:9.1f}{lp:8.1f}")
    print("  " + "─" * (W - 2))

    # ── 逐类每帧平均检测数对比 ────────────────────────────────────────────────
    print(f"\n  逐类 每帧平均检测数")
    print("  " + "─" * (W - 2))
    hdr = f"  {'配置':48s}" + "".join(f" {LABEL_NAMES[c]:>11s}" for c in MAIN_LABELS)
    print(hdr)
    print("  " + "─" * (W - 2))
    for name, lm, lp, avg, td, _ in summary:
        row = f"  {name:48s}"
        for c in MAIN_LABELS:
            row += f" {td[c]/n_img:10.2f} "
        print(row)
    print("  " + "─" * (W - 2))

    # ── 方案汇总说明 ──────────────────────────────────────────────────────────
    if summary:
        best_spd = min(summary, key=lambda x: x[1])
        most_det = max(summary, key=lambda x: x[3])
        print(f"\n  ★ 最快推理   : {best_spd[0].strip()}  →  {best_spd[1]:.1f}ms")
        print(f"  ★ 最多检测数  : {most_det[0].strip()}  →  每帧 {most_det[3]:.2f} 个")
        print(f"  ★ 可视化样本  : {vis_dir}")

    # ── 关闭引擎 ──────────────────────────────────────────────────────────────
    for inf in [inf375, inf500, inf500i, inf700]:
        if inf: inf.close()

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out = WS / "test_images/compare_results/eval_nuimages_summary.json"
    out.parent.mkdir(exist_ok=True)
    save = [
        {
            "config":       n,
            "lat_mean_ms":  lm,
            "lat_p95_ms":   lp,
            "avg_dets_per_img": avg,
            "total_per_class": {LABEL_NAMES.get(c, str(c)): td[c]
                                 for c in range(1, 11)},
            "avg_per_class":   {LABEL_NAMES.get(c, str(c)): td[c] / n_img
                                 for c in range(1, 11)},
        }
        for n, lm, lp, avg, td, _ in summary
    ]
    json.dump(save, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
