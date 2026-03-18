#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_presentation.py ── Faster R-CNN ROS2 项目一键演示脚本
═══════════════════════════════════════════════════════════════
用法：
    python3 demo_presentation.py                   # 自动运行全部演示
    python3 demo_presentation.py --stage 2         # 只运行指定阶段
    python3 demo_presentation.py --no-infer       # 跳过推理，只展示已有结果
    python3 demo_presentation.py --images 3       # 实时推理的图片数量（默认6）
    python3 demo_presentation.py --output demo_report.html  # 指定报告路径
"""

import argparse
import base64
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

# ─────────────────────────── ANSI 颜色 ─────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    BG_BLU = "\033[44m"
    BG_GRN = "\033[42m"
    BG_YEL = "\033[43m"

def p(text=""):  print(text)
def ph(text):    print(f"{C.BOLD}{C.CYAN}{text}{C.RESET}")
def ps(text):    print(f"  {C.GREEN}✓{C.RESET}  {text}")
def pw(text):    print(f"  {C.YELLOW}⚠{C.RESET}  {text}")
def pe(text):    print(f"  {C.RED}✗{C.RESET}  {text}")
def pi(text):    print(f"  {C.BLUE}ℹ{C.RESET}  {text}")
def pbar(text):  print(f"\n{C.BG_BLU}{C.WHITE}{C.BOLD}  {text:<74}  {C.RESET}\n")

def progress_bar(current, total, width=50, label=""):
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  [{C.GREEN}{bar}{C.RESET}] {current:3d}/{total}  {label:<20}", end="", flush=True)

def separator(char="─", width=74):
    print(f"{C.BLUE}  {''.join([char]*width)}{C.RESET}")

# ─────────────────────────── 路径配置 ──────────────────────────────────────
WS           = Path("/home/nvidia/ros2_ws")
INSTALL      = WS / "install/faster_rcnn_ros/share/faster_rcnn_ros"
MODELS       = INSTALL / "models"
ENGINE_FP16  = MODELS / "faster_rcnn_500.engine"
ENGINE_INT8  = MODELS / "faster_rcnn_500_int8.engine"
LABELS_FILE  = MODELS / "labels.txt"
KITTI_IMGS   = WS / "test_images/kitti_100/images"
KITTI_LBLS   = WS / "test_images/kitti_100/labels"
RESULTS_BASE = WS / "test_images/compare_results"
GT_JSON      = RESULTS_BASE / "gt_eval_summary.json"

INPUT_H, INPUT_W = 500, 1242
MAX_DET = 2000

# 演示用精选图片（FP16+0.5 检测框最多的前6张）
DEMO_IMAGES = ["000049", "000008", "000099", "000053", "000089", "000025"]

# ─────────────────────────── CUDA / TRT ─────────────────────────────────────
_cudart = ctypes.cdll.LoadLibrary("libcudart.so")

def _malloc(n):
    p = ctypes.c_void_p()
    assert _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    return p.value

def _free(p):   _cudart.cudaFree(ctypes.c_void_p(p))
def _h2d(d, a): _cudart.cudaMemcpy(ctypes.c_void_p(d), a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes), ctypes.c_int(1))
def _d2h(a, s): _cudart.cudaMemcpy(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(s), ctypes.c_size_t(a.nbytes), ctypes.c_int(2))


class Inferencer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path, "rb").read())
        self._ctx = self._eng.create_execution_context()
        self._ctx.set_input_shape("image", (1, 3, INPUT_H, INPUT_W))
        self._dinp    = _malloc(3 * INPUT_H * INPUT_W * 4)
        self._dscores = _malloc(MAX_DET * 4)
        self._dlabels = _malloc(MAX_DET * 4)
        self._dboxes  = _malloc(MAX_DET * 4 * 4)
        self._ctx.set_tensor_address("image",  self._dinp)
        self._ctx.set_tensor_address("scores", self._dscores)
        self._ctx.set_tensor_address("labels", self._dlabels)
        self._ctx.set_tensor_address("boxes",  self._dboxes)

    def infer(self, bgr, threshold=0.5):
        blob = cv2.dnn.blobFromImage(bgr, 1.0/255.0, (INPUT_W, INPUT_H), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        _h2d(self._dinp, blob)
        t0 = time.perf_counter()
        self._ctx.execute_async_v3(0)
        _cudart.cudaDeviceSynchronize()
        ms = (time.perf_counter() - t0) * 1000

        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n <= 0:
            return [], ms
        h_sc = np.empty(n,     dtype=np.float32); _d2h(h_sc, self._dscores)
        h_lb = np.empty(n,     dtype=np.int32);   _d2h(h_lb, self._dlabels)
        h_bx = np.empty(n * 4, dtype=np.float32); _d2h(h_bx, self._dboxes)
        h_bx = h_bx.reshape(n, 4)

        oh, ow = bgr.shape[:2]
        sx, sy = ow / INPUT_W, oh / INPUT_H
        dets = []
        for i in range(n):
            if h_sc[i] < threshold: continue
            x1 = float(h_bx[i,0]*sx); y1 = float(h_bx[i,1]*sy)
            x2 = float(h_bx[i,2]*sx); y2 = float(h_bx[i,3]*sy)
            if x2 <= x1 or y2 <= y1: continue
            dets.append({"score": float(h_sc[i]), "label": int(h_lb[i]),
                         "box": [int(x1), int(y1), int(x2), int(y2)]})
        return dets, ms

    def warmup(self, dummy):
        for _ in range(5): self.infer(dummy, 0.1)

    def close(self):
        _free(self._dinp); _free(self._dscores); _free(self._dlabels); _free(self._dboxes)


# ─────────────────────────── 工具函数 ──────────────────────────────────────
CLASS_NAMES = LABELS_FILE.read_text().splitlines()

# 类别颜色（BGR）
CLASS_COLORS = {
    1: (0, 200, 0),    # car - 绿
    2: (0, 255, 128),  # truck - 浅绿
    3: (0, 180, 255),  # bus - 橙
    4: (255, 128, 0),  # trailer - 蓝橙
    5: (128, 0, 255),  # construction_vehicle - 紫
    6: (0, 0, 255),    # pedestrian - 红
    7: (255, 0, 128),  # motorcycle - 粉
    8: (255, 0, 0),    # bicycle - 蓝
    9: (0, 255, 255),  # traffic_cone - 黄
    10:(128, 128, 128),# barrier - 灰
}

def draw_dets(img, dets, show_score=True):
    out = img.copy()
    for d in dets:
        cls = d["label"]
        color = CLASS_COLORS.get(cls, (0, 255, 0))
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        txt = f"{name}: {d['score']:.2f}" if show_score else name
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(out, txt, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return out


def make_side_by_side(orig, result, label_l="原始图像", label_r="检测结果", scale=0.6):
    """将原始图和结果图横向拼接，缩放后返回"""
    h, w = orig.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    lo = cv2.resize(orig,   (nw, nh))
    lr = cv2.resize(result, (nw, nh))

    bar_h = 30
    canvas = np.zeros((nh + bar_h, nw * 2 + 6, 3), dtype=np.uint8)
    canvas[:, :, :] = (40, 40, 40)   # 深灰背景

    canvas[bar_h:, :nw]       = lo
    canvas[bar_h:, nw+6:]     = lr

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, label_l, (10, 22),       font, 0.65, (200,200,200), 1)
    cv2.putText(canvas, label_r, (nw+16, 22),    font, 0.65, (100,255,100), 1)
    return canvas


def img_to_b64(img, ext=".jpg", q=85):
    _, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()

# ─────────────────── Stage 0：欢迎横幅 + 环境信息 ──────────────────────────

def stage_env():
    p()
    print(f"{C.BOLD}{C.CYAN}{'═'*78}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  ██████  Faster R-CNN ROS2 目标检测系统 ── 培训演示{' '*18}██████{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  Jetson AGX Orin  ·  TensorRT FP16/INT8  ·  ROS2 Foxy{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═'*78}{C.RESET}")
    p()

    ph("【阶段 0】环境信息检查")
    separator()
    p()

    # Python / TRT / OpenCV / CUDA
    import platform
    pi(f"操作系统  : {platform.platform()}")
    pi(f"Python    : {sys.version.split()[0]}")
    pi(f"OpenCV    : {cv2.__version__}")
    pi(f"TensorRT  : {trt.__version__}")

    # CUDA 设备
    try:
        n_dev = ctypes.c_int(0)
        _cudart.cudaGetDeviceCount(ctypes.byref(n_dev))
        pi(f"CUDA 设备 : {n_dev.value} 个")
    except Exception:
        pw("无法查询 CUDA 设备")

    # ROS2
    ros_found = Path("/opt/ros/foxy/bin/ros2").exists() or Path("/opt/ros/humble/bin/ros2").exists()
    if ros_found:
        ps("ROS2 环境  : 已安装")
    else:
        pw("ROS2 环境  : 未检测到（功能受限）")

    p()
    # 引擎文件检查
    pi("引擎文件检查：")
    engines = [
        (ENGINE_FP16, "FasterRCNN FP16  500×1242"),
        (ENGINE_INT8, "FasterRCNN INT8  500×1242"),
    ]
    all_ok = True
    for path, desc in engines:
        if path.exists():
            mb = path.stat().st_size // (1024**2)
            ps(f"{desc:<38} {mb:>4} MB   {path.name}")
        else:
            pe(f"{desc:<38} 未找到！ ({path})")
            all_ok = False

    if not all_ok:
        pe("\n  关键引擎文件缺失，无法继续演示。")
        pe("  请先运行：python3 src/faster_rcnn_ros/models/build_engine.py --height 500 --width 1242")
        sys.exit(1)

    p()
    # 测试数据
    img_count = len(list(KITTI_IMGS.glob("*.png")))
    lbl_count = len(list(KITTI_LBLS.glob("*.txt")))
    ps(f"测试图片   : {img_count} 张  ({KITTI_IMGS})")
    ps(f"GT 标注    : {lbl_count} 个  ({KITTI_LBLS})")

    p()
    print(f"  {C.YELLOW}{'─'*72}{C.RESET}")
    print(f"  {C.YELLOW}演示将分 4 个阶段进行，预计总耗时 < 2 分钟{C.RESET}")
    print(f"  {C.YELLOW}{'─'*72}{C.RESET}")
    p()


# ─────────────────── Stage 1：架构讲解 ────────────────────────────────────

def stage_architecture():
    pbar("阶段 1：项目架构")

    arch = """
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  用户输入（图片 / 视频 / 相机话题）                   │
  └─────────────────────────┬───────────────────────────────────────────┘
                             │
  ┌─────────────────────────▼───────────────────────────────────────────┐
  │           ROS2 节点 ── faster_rcnn_node.cpp                         │
  │   ● 话题模式：订阅 /detectnet/image_in → 发布 /detectnet/overlay    │
  │   ● 文件模式：单图 / 视频 / 批量目录                                │
  │   ● 支持双引擎多尺度 + 类别专属置信度阈值                           │
  └─────────────────────────┬───────────────────────────────────────────┘
                             │ cv::Mat BGR
  ┌─────────────────────────▼───────────────────────────────────────────┐
  │           检测器封装 ── faster_rcnn_detector.cpp                    │
  │   blobFromImage (resize 500×1242, /255, RGB)                        │
  │   ─→ cudaMemcpyAsync (H2D, ~3.7 MB)                                 │
  │   ─→ enqueueV3 (GPU 推理, ~63-71 ms)                                │
  │   ─→ cudaStreamSynchronize                                           │
  │   ─→ cudaMemcpy (D2H, scores/labels/boxes)                          │
  │   ─→ 坐标缩放 + 阈值过滤 → vector<Detection>                        │
  └─────────────────────────┬───────────────────────────────────────────┘
                             │
  ┌─────────────────────────▼───────────────────────────────────────────┐
  │           TensorRT 推理引擎 (FP16 / INT8)                           │
  │   image[1,3,500,1242] → FPN → RPN → ROI Align → Head               │
  │   输出: scores[-1]  labels[-1]  boxes[-1,4]                         │
  └─────────────────────────────────────────────────────────────────────┘"""
    print(arch)

    p()
    ph("  Faster R-CNN 两阶段检测流程：")
    steps = [
        ("  Stage 1: RPN（区域提议网络）", "扫描所有位置，生成 ~5000 个候选框"),
        ("  Stage 2: ROI Head（感兴趣区域头）", "对每个候选框精细分类+坐标回归"),
        ("  NMS（非极大值抑制）",          "去除重叠度>50%的冗余框"),
        ("  阈值过滤",                     "保留置信度>threshold 的检测结果"),
    ]
    for name, desc in steps:
        print(f"  {C.GREEN}▶{C.RESET} {C.BOLD}{name:<42}{C.RESET}  {desc}")
    p()


# ─────────────────── Stage 2：实时推理演示 ─────────────────────────────────

def stage_inference(n_images=6, threshold=0.5):
    pbar("阶段 2：实时推理演示（FP16 引擎）")

    ph("  加载 FP16 引擎...")
    t_load = time.perf_counter()
    inf = Inferencer(str(ENGINE_FP16))
    pi(f"  引擎加载耗时: {(time.perf_counter()-t_load)*1000:.0f} ms")
    p()

    ph("  Warmup（预热 5 次）...")
    dummy = cv2.imread(str(KITTI_IMGS / (DEMO_IMAGES[0] + ".png")))
    inf.warmup(dummy)
    ps("  预热完成")
    p()

    ph(f"  开始推理演示图片（阈值={threshold}，共 {n_images} 张）")
    separator()
    p()

    results = []
    images_to_use = DEMO_IMAGES[:n_images]

    for idx, stem in enumerate(images_to_use):
        img_path = KITTI_IMGS / (stem + ".png")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            pw(f"  跳过 {stem}（文件不存在）")
            continue

        dets, ms = inf.infer(bgr, threshold)

        # 类别统计
        cls_cnt = {}
        for d in dets:
            n = CLASS_NAMES[d["label"]] if d["label"] < len(CLASS_NAMES) else str(d["label"])
            cls_cnt[n] = cls_cnt.get(n, 0) + 1
        cls_str = "  ".join(f"{k}×{v}" for k, v in sorted(cls_cnt.items()))

        ms_bar = "█" * min(int(ms / 3), 30)
        ms_color = C.GREEN if ms < 70 else (C.YELLOW if ms < 90 else C.RED)
        print(f"  {C.WHITE}{stem}.png{C.RESET}  "
              f"{ms_color}{ms:6.1f}ms{C.RESET} [{ms_color}{ms_bar:<30}{C.RESET}]  "
              f"{C.BOLD}{len(dets):2d}个目标{C.RESET}  {C.CYAN}{cls_str}{C.RESET}")

        overlay = draw_dets(bgr, dets)
        results.append({"stem": stem, "bgr": bgr, "overlay": overlay,
                         "dets": dets, "ms": ms})

    inf.close()

    p()
    if results:
        lat = [r["ms"] for r in results]
        print(f"  {'─'*60}")
        print(f"  推理统计：均值={np.mean(lat):.1f}ms  "
              f"中位={np.median(lat):.1f}ms  "
              f"P95={np.percentile(lat,95):.1f}ms")
    p()
    return results


# ─────────────────── Stage 3：精度对比表 ───────────────────────────────────

def stage_accuracy():
    pbar("阶段 3：精度对比（100张KITTI测试集）")

    if not GT_JSON.exists():
        pw(f"  未找到评估结果 JSON: {GT_JSON}")
        pw("  请先运行: python3 gt_eval_all.py")
        return []

    data = json.loads(GT_JSON.read_text())
    p()

    # 表头
    cols = [("配置",20), ("引擎",6), ("GT",5), ("Pred",6), ("Hit",5),
            ("Miss",6), ("FP",5), ("Recall",8), ("Prec",8),
            ("延迟均值",10), ("P95",8)]
    hdr = "".join(f"{c:>{w}} " for c, w in cols)
    sep = "".join(f"{'─'*w} " for _, w in cols)

    ph("  " + hdr)
    print(f"  {C.BLUE}{sep}{C.RESET}")

    best_rec  = max(data, key=lambda x: x["Rec%"])
    best_lat  = min(data, key=lambda x: x["lat_mean"])

    for r in data:
        rec_color  = C.GREEN  if r["Rec%"] >= 70 else (C.YELLOW if r["Rec%"] >= 50 else C.RED)
        prec_color = C.GREEN  if r["Prec%"] >= 70 else (C.YELLOW if r["Prec%"] >= 50 else C.RED)
        lat_color  = C.GREEN  if r["lat_mean"] < 65 else C.YELLOW

        def fmt(v, w, color=C.RESET, decimals=None):
            if decimals is not None:
                s = f"{v:>{w}.{decimals}f}"
            elif isinstance(v, float):
                s = f"{v:>{w}.1f}"
            elif isinstance(v, int):
                s = f"{v:>{w}d}"
            else:
                s = f"{v:>{w}}"
            return f"{color}{s}{C.RESET}"

        tag = ""
        if r is best_rec and r is best_lat:
            tag = f" {C.BG_GRN}{C.WHITE} ★最优 {C.RESET}"
        elif r is best_rec:
            tag = f" {C.YELLOW}↑Recall{C.RESET}"
        elif r is best_lat:
            tag = f" {C.CYAN}↑Speed{C.RESET}"

        row = (f"  {fmt(r['cfg'],20)}  "
               f"{fmt(r['engine_mb'],4)}MB  "
               f"{fmt(r['GT'],5)}  "
               f"{fmt(r['Pred'],5)}  "
               f"{fmt(r['Hit'],5)}  "
               f"{fmt(r['Miss'],5)}  "
               f"{fmt(r['FP'],5)}  "
               f"{fmt(r['Rec%'],6,rec_color,1)}%  "
               f"{fmt(r['Prec%'],6,prec_color,1)}%  "
               f"{fmt(r['lat_mean'],8,lat_color,1)}ms  "
               f"{fmt(r['lat_p95'],6,C.RESET,1)}ms"
               f"{tag}")
        print(row)

    print(f"  {C.BLUE}{sep}{C.RESET}")
    p()

    # 关键结论
    ph("  关键结论：")
    conclusions = [
        (C.GREEN,  "INT8 vs FP16（阈值=0.5）",
                   "Recall 损失仅 -1.2%，延迟快 8.5ms，引擎体积减半（114→58MB）→ 推荐方案"),
        (C.YELLOW, "阈值 0.5 → 0.25",
                   "Recall 提升 +23%，但误检（FP）暴增 8倍，Precision 从73%跌至32%"),
        (C.CYAN,   "为何推理 >60ms？",
                   "Faster R-CNN 是两阶段架构，Stage1(RPN)+Stage2(ROI Head)串行，无法<10ms"),
    ]
    for color, title, detail in conclusions:
        print(f"  {color}▶ {C.BOLD}{title}{C.RESET}")
        print(f"    {detail}")
        p()

    return data


# ─────────────────── Stage 4：可视化 Mosaic ─────────────────────────────────

def stage_visualize(live_results):
    pbar("阶段 4：检测结果可视化")

    # 优先使用实时推理结果，否则使用已保存的结果图
    show_imgs = []

    if live_results:
        ph("  使用刚才推理的实时结果生成对比图...")
        for r in live_results[:6]:
            show_imgs.append((r["bgr"], r["overlay"],
                              f"{r['stem']}  {len(r['dets'])}个目标  {r['ms']:.1f}ms"))
    else:
        ph("  使用预计算的结果图片（FP16 + thr=0.5）...")
        for stem in DEMO_IMAGES:
            orig = cv2.imread(str(KITTI_IMGS / (stem + ".png")))
            res  = cv2.imread(str(RESULTS_BASE / "FP16_thr0.5" / f"result_{stem}.png"))
            if orig is None or res is None: continue
            show_imgs.append((orig, res, f"{stem}"))

    if not show_imgs:
        pw("  无可用图片，跳过可视化")
        return None

    # 生成单个大图 mosaic（3列×N行）
    side_imgs = []
    for orig, res, title in show_imgs:
        panel = make_side_by_side(orig, res, label_r=title, scale=0.5)
        side_imgs.append(panel)

    # 排成3列布局
    cols = 3
    rows = (len(side_imgs) + cols - 1) // cols
    H, W = side_imgs[0].shape[:2]

    mosaic = np.zeros((H * rows, W * cols, 3), dtype=np.uint8)
    mosaic[:] = (30, 30, 30)

    for i, img in enumerate(side_imgs):
        r, c = i // cols, i % cols
        mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = img

    out_path = WS / "demo_mosaic.jpg"
    cv2.imwrite(str(out_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ps(f"  可视化结果已保存: {out_path}")

    # 尝试显示（如果有 DISPLAY 环境变量）
    if os.environ.get("DISPLAY"):
        import sys as _sys
        # 静默 cv2 Qt 字体告警（C-level，需 fd 重定向）
        old_fd2 = os.dup(2)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            cv2.imshow("Faster R-CNN 检测结果对比", mosaic)
            os.dup2(old_fd2, 2)
        os.close(old_fd2)
        pi("  按任意键继续（或等待5秒自动跳过）...")
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    else:
        pi("  无图形显示环境，结果已保存到文件")

    p()
    return mosaic


# ─────────────────── HTML 报告生成 ─────────────────────────────────────────

def generate_html(live_results, accuracy_data, mosaic, output_path):
    pbar("生成 HTML 培训报告")

    mosaic_b64 = img_to_b64(mosaic) if mosaic is not None else ""

    # 准备结果图 base64
    gallery_html = ""
    items = live_results if live_results else []
    for r in items[:6]:
        side = make_side_by_side(r["bgr"], r["overlay"],
                                 label_l="原图",
                                 label_r=f"{len(r['dets'])}目标 {r['ms']:.1f}ms",
                                 scale=0.55)
        b = img_to_b64(side)
        gallery_html += f"""
        <div class="gallery-item">
          <img src="data:image/jpeg;base64,{b}" alt="{r['stem']}">
          <div class="caption">{r['stem']} · {len(r['dets'])} 个目标 · {r['ms']:.1f}ms</div>
        </div>"""

    # 精度表格
    table_html = ""
    if accuracy_data:
        for r in accuracy_data:
            rec_cls   = "good" if r["Rec%"] >= 70 else ("warn" if r["Rec%"] >= 50 else "bad")
            prec_cls  = "good" if r["Prec%"] >= 70 else ("warn" if r["Prec%"] >= 50 else "bad")
            lat_cls   = "good" if r["lat_mean"] < 65 else "warn"
            table_html += f"""
        <tr>
          <td><b>{r['cfg']}</b></td>
          <td>{r['engine_mb']} MB</td>
          <td>{r['GT']}</td>
          <td>{r['Pred']}</td>
          <td>{r['Hit']}</td>
          <td>{r['Miss']}</td>
          <td>{r['FP']}</td>
          <td class="{rec_cls}">{r['Rec%']:.1f}%</td>
          <td class="{prec_cls}">{r['Prec%']:.1f}%</td>
          <td class="{lat_cls}">{r['lat_mean']:.1f}ms</td>
          <td>{r['lat_p95']:.1f}ms</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-cn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Faster R-CNN ROS2 项目演示报告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", sans-serif; background: #1a1a2e; color: #e0e0e0; }}
  .header {{ background: linear-gradient(135deg, #0f3460, #16213e);
             padding: 40px; text-align: center; border-bottom: 3px solid #e94560; }}
  .header h1 {{ font-size: 2em; color: #e94560; margin-bottom: 10px; }}
  .header p  {{ color: #a0c4ff; font-size: 1.05em; }}
  .section  {{ margin: 30px auto; max-width: 1200px; padding: 0 20px; }}
  h2 {{ color: #e94560; border-bottom: 2px solid #e94560; padding-bottom: 8px;
        margin-bottom: 20px; font-size: 1.4em; }}
  .card {{ background: #16213e; border-radius: 10px; padding: 20px;
           border: 1px solid #0f3460; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{ background: #0f3460; color: #a0c4ff; padding: 10px;
        text-align: left; border: 1px solid #1a4080; }}
  td {{ padding: 9px 10px; border: 1px solid #1a4080; }}
  tr:nth-child(even) {{ background: #0d1b3e; }}
  tr:hover {{ background: #1e3a6e; }}
  .good {{ color: #4caf50; font-weight: bold; }}
  .warn {{ color: #ff9800; font-weight: bold; }}
  .bad  {{ color: #f44336; font-weight: bold; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 15px; }}
  .gallery-item {{ flex: 1; min-width: 300px; }}
  .gallery-item img {{ width: 100%; border-radius: 6px;
                        border: 2px solid #0f3460; }}
  .caption {{ font-size: 0.8em; color: #a0c4ff; text-align: center;
              margin-top: 6px; }}
  .mosaic img {{ width: 100%; border-radius: 8px; border: 2px solid #e94560; }}
  .info-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
  .info-item {{ background: #0d1b3e; border-radius: 8px; padding: 15px;
                border-left: 4px solid #e94560; }}
  .info-item .label {{ color: #a0c4ff; font-size: 0.85em; }}
  .info-item .value {{ font-size: 1.3em; font-weight: bold; color: #4caf50; }}
  footer {{ text-align: center; padding: 30px; color: #555;
            border-top: 1px solid #0f3460; margin-top: 40px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🚗 Faster R-CNN ROS2 目标检测系统</h1>
  <p>平台：Jetson AGX Orin · TensorRT 8.5 · ROS2 Foxy · 生成时间：{time.strftime('%Y-%m-%d %H:%M')}</p>
</div>

<div class="section">
  <h2>📊 系统性能总览</h2>
  <div class="card">
    <div class="info-grid">
      <div class="info-item">
        <div class="label">推理引擎</div>
        <div class="value">TensorRT FP16</div>
      </div>
      <div class="info-item">
        <div class="label">平均推理延迟</div>
        <div class="value">~63–71 ms</div>
      </div>
      <div class="info-item">
        <div class="label">召回率（KITTI，IoU≥0.5）</div>
        <div class="value">49–73%</div>
      </div>
      <div class="info-item">
        <div class="label">引擎大小</div>
        <div class="value">58–114 MB</div>
      </div>
      <div class="info-item">
        <div class="label">检测类别</div>
        <div class="value">10 类</div>
      </div>
      <div class="info-item">
        <div class="label">输入分辨率</div>
        <div class="value">500 × 1242</div>
      </div>
    </div>
  </div>
</div>

<div class="section">
  <h2>🏗️ 系统架构</h2>
  <div class="card">
    <pre style="color:#a0c4ff; font-size:0.85em; line-height:1.7">
用户输入（图片 / 视频 / 相机话题）
    │
    ▼
ROS2 节点（faster_rcnn_node.cpp）
├─ 话题模式：订阅 /detectnet/image_in → 发布 /detectnet/overlay
├─ 文件模式：单图 / 视频 / 批量目录
└─ 双引擎多尺度 + 类别专属置信度阈值
    │
    ▼
检测器封装（faster_rcnn_detector.cpp）
├─ blobFromImage：resize 500×1242, /255, BGR→RGB
├─ cudaMemcpyAsync H2D (~3.7 MB)
├─ TRT enqueueV3 (GPU 推理: ~63-71ms)
└─ D2H + 坐标缩放 + 阈值过滤
    │
    ▼
TensorRT 引擎（FP16 / INT8）
image[1,3,500,1242] → FPN → RPN → ROI Align → Head
    └─→ scores[-1]  labels[-1]  boxes[-1,4]
    </pre>
  </div>
</div>

<div class="section">
  <h2>📈 四配置综合指标对比（KITTI 100张，IoU≥0.5）</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>配置</th><th>引擎</th><th>GT</th><th>Pred</th>
          <th>Hit</th><th>Miss</th><th>FP</th>
          <th>Recall</th><th>Precision</th>
          <th>延迟均值</th><th>延迟P95</th>
        </tr>
      </thead>
      <tbody>{table_html}</tbody>
    </table>
  </div>
  <div class="card">
    <strong style="color:#e94560">核心结论：</strong>
    <ul style="margin-top:10px; padding-left:20px; line-height:2">
      <li>✅ <b>INT8 vs FP16（阈值=0.5）</b>：Recall 损失仅 −1.2%，延迟快 8.5ms，引擎减半 → 推荐方案</li>
      <li>⚠️ <b>阈值降至 0.25</b>：Recall 提升 +23%，但误检(FP)暴增 8 倍，Precision 73%→32%</li>
      <li>ℹ️ <b>为何 >60ms</b>：两阶段架构（RPN+ROI Head 串行），无法达到 YOLO 的单阶段 3ms</li>
    </ul>
  </div>
</div>

{"" if not gallery_html else f'''
<div class="section">
  <h2>🔍 实时推理结果（FP16 + thr=0.5）</h2>
  <div class="card">
    <div class="gallery">{gallery_html}</div>
  </div>
</div>
'''}

{"" if not mosaic_b64 else f'''
<div class="section">
  <h2>🖼️ 检测结果全览（原图 vs 结果图）</h2>
  <div class="card mosaic">
    <img src="data:image/jpeg;base64,{mosaic_b64}" alt="检测结果 Mosaic">
  </div>
</div>
'''}

<div class="section">
  <h2>📚 快速参考</h2>
  <div class="card">
    <table>
      <thead><tr><th>操作</th><th>命令</th></tr></thead>
      <tbody>
        <tr><td>编译</td><td><code>colcon build --packages-select faster_rcnn_ros &amp;&amp; source install/setup.bash</code></td></tr>
        <tr><td>GPU 锁频</td><td><code>sudo jetson_clocks</code></td></tr>
        <tr><td>批量推理</td><td><code>ros2 launch faster_rcnn_ros faster_rcnn.launch.xml input_path:=./test_images output_path:=./results</code></td></tr>
        <tr><td>精度评估</td><td><code>python3 gt_eval_all.py</code></td></tr>
        <tr><td>一键演示</td><td><code>bash demo.sh</code> 或 <code>python3 demo_presentation.py</code></td></tr>
      </tbody>
    </table>
  </div>
</div>

<footer>Faster R-CNN ROS2 项目 · Jetson AGX Orin · {time.strftime('%Y-%m-%d')}</footer>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    ps(f"  HTML 报告已生成: {output_path}")
    pi(f"  在浏览器中打开: file://{Path(output_path).resolve()}")
    p()


# ─────────────────────────── 主函数 ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Faster R-CNN ROS2 一键培训演示脚本")
    parser.add_argument("--stage", type=int, default=0,
                        help="只运行指定阶段 (0=全部, 1=架构, 2=推理, 3=精度, 4=可视化)")
    parser.add_argument("--no-infer", action="store_true",
                        help="跳过实时 TRT 推理，仅展示已有结果")
    parser.add_argument("--images", type=int, default=6,
                        help="实时推理的图片数量 (默认 6)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="检测置信度阈值 (默认 0.5)")
    parser.add_argument("--output", type=str, default="demo_report.html",
                        help="HTML 报告输出路径 (默认 demo_report.html)")
    args = parser.parse_args()

    only = args.stage

    try:
        # 阶段 0：环境检查（始终运行）
        stage_env()

        # 阶段 1：架构讲解
        if only in (0, 1):
            stage_architecture()
            if only == 1: return

        # 阶段 2：实时推理
        live_results = []
        if only in (0, 2):
            if args.no_infer:
                pbar("阶段 2：跳过实时推理（--no-infer）")
                p()
            else:
                live_results = stage_inference(args.images, args.threshold)
            if only == 2: return

        # 阶段 3：精度对比
        accuracy_data = []
        if only in (0, 3):
            accuracy_data = stage_accuracy()
            if only == 3: return

        # 阶段 4：可视化
        mosaic = None
        if only in (0, 4):
            mosaic = stage_visualize(live_results)
            if only == 4: return

        # HTML 报告（仅全流程时生成）
        if only == 0:
            generate_html(live_results, accuracy_data, mosaic,
                          str(Path(args.output).resolve()))

        print(f"\n{C.BG_GRN}{C.WHITE}{C.BOLD}  ✓ 演示完成！{'─'*58}  {C.RESET}")
        p()

    except KeyboardInterrupt:
        p()
        pw("演示被用户中断")
    except Exception as e:
        pe(f"演示出错: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
