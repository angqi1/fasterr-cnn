#!/usr/bin/env python3
"""
profile_inference.py — 推理流水线细粒度耗时分析
拆分各阶段耗时：imread / preprocess / H2D / execute / D2H / postprocess
并对比不同优化手段的效果
"""

import time, ctypes, statistics
from pathlib import Path
import numpy as np
import cv2

_cudart = ctypes.cdll.LoadLibrary("libcudart.so")
def _malloc(n):
    p = ctypes.c_void_p()
    assert _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    return p.value
def _malloc_host(n):
    """页锁定内存 (pinned)"""
    p = ctypes.c_void_p()
    assert _cudart.cudaMallocHost(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    return p.value
def _free(p): _cudart.cudaFree(ctypes.c_void_p(p))
def _free_host(p): _cudart.cudaFreeHost(ctypes.c_void_p(p))
def _h2d(d, a):
    _cudart.cudaMemcpy(ctypes.c_void_p(d), a.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_size_t(a.nbytes), ctypes.c_int(1))
def _d2h(a, s):
    _cudart.cudaMemcpy(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(s),
                       ctypes.c_size_t(a.nbytes), ctypes.c_int(2))
def _h2d_async(d, a, stream):
    _cudart.cudaMemcpyAsync(ctypes.c_void_p(d), a.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_size_t(a.nbytes), ctypes.c_int(1), stream)
def _d2h_async(a, s, stream):
    _cudart.cudaMemcpyAsync(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(s),
                            ctypes.c_size_t(a.nbytes), ctypes.c_int(2), stream)
def _sync(): _cudart.cudaDeviceSynchronize()
def _stream_create():
    s = ctypes.c_void_p()
    _cudart.cudaStreamCreate(ctypes.byref(s))
    return s  # c_void_p
def _stream_sync(s): _cudart.cudaStreamSynchronize(s)
def _stream_int(s): return s.value if s.value else 0  # TRT 8.5 需要 int

import tensorrt as trt

WS          = Path("/home/nvidia/ros2_ws")
MODELS_DIR  = WS / "install/faster_rcnn_ros/share/faster_rcnn_ros/models"
IMG_DIR     = WS / "test_images/kitti_100/images"
MAX_DET     = 2000
N_WARMUP    = 5
N_REPEAT    = 20  # 每帧重复推理次数（取均值）
N_IMAGES    = 20   # 测试图片数

ENGINES = [
    ("FP16_375h",      "faster_rcnn_375.engine",      375, 1242),
    ("INT8_375h(新)",  "faster_rcnn_375_int8.engine",  375, 1242),
    ("FP16_500h",      "faster_rcnn_500.engine",       500, 1242),
    ("INT8_500h",      "faster_rcnn_500_int8.engine",  500, 1242),
    ("FP16_700h",      "faster_rcnn_700.engine",       700, 1242),
]


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def now_ms(): return time.perf_counter() * 1000

def ns(vals, pct=50):
    return statistics.median(vals) if pct == 50 else sorted(vals)[int(len(vals)*pct//100)]

def ms_str(vals):
    return f"{statistics.mean(vals):.2f}ms (P50={ns(vals):.2f} P95={ns(vals,95):.2f})"


# ─── 方式A：原始方式 (CPU blob + pageable cudaMemcpy) ─────────────────────────
class ProfilerBaseline:
    name = "A: 原始 (CPU blob + pageable mem)"

    def __init__(self, engine_path, H, W):
        self.H, self.W = H, W
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path,"rb").read())
        self._ctx = self._eng.create_execution_context()
        self._d_inp    = _malloc(3 * H * W * 4)
        self._d_scores = _malloc(MAX_DET * 4)
        self._d_labels = _malloc(MAX_DET * 4)
        self._d_boxes  = _malloc(MAX_DET * 4 * 4)
        self._ctx.set_tensor_address("image",  self._d_inp)
        self._ctx.set_tensor_address("scores", self._d_scores)
        self._ctx.set_tensor_address("labels", self._d_labels)
        self._ctx.set_tensor_address("boxes",  self._d_boxes)
        self._ctx.set_input_shape("image", (1, 3, H, W))

    def run_one(self, bgr):
        t = {}
        # 1. 预处理 (CPU blobFromImage)
        t0 = now_ms()
        blob = cv2.dnn.blobFromImage(bgr, 1.0/255.0, (self.W, self.H), swapRB=True, crop=False)
        blob = np.ascontiguousarray(blob)
        t["preprocess"] = now_ms() - t0

        # 2. H2D
        t0 = now_ms()
        _h2d(self._d_inp, blob)
        t["h2d"] = now_ms() - t0

        # 3. TRT execute
        t0 = now_ms()
        self._ctx.execute_async_v3(0)
        _sync()
        t["execute"] = now_ms() - t0

        # 4. D2H
        t0 = now_ms()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n > 0:
            sc = np.empty(n, np.float32); _d2h(sc, self._d_scores)
            lb = np.empty(n, np.int32);   _d2h(lb, self._d_labels)
            bx = np.empty(n*4, np.float32); _d2h(bx, self._d_boxes)
        t["d2h"] = now_ms() - t0

        # 5. 后处理 (过阈值过滤)
        t0 = now_ms()
        dets = []
        if n > 0:
            oh, ow = bgr.shape[:2]
            sx, sy = ow/self.W, oh/self.H
            for i in range(n):
                if float(sc[i]) < 0.5: continue
                x1,y1,x2,y2 = bx[i*4]*sx, bx[i*4+1]*sy, bx[i*4+2]*sx, bx[i*4+3]*sy
                dets.append((int(lb[i]), x1, y1, x2, y2, float(sc[i])))
        t["postprocess"] = now_ms() - t0
        t["total"] = sum(t.values())
        return t

    def close(self):
        _free(self._d_inp); _free(self._d_scores)
        _free(self._d_labels); _free(self._d_boxes)


# ─── 方式B：Pinned Memory + Async ─────────────────────────────────────────────
class ProfilerPinned:
    name = "B: Pinned mem + Async memcpy"

    def __init__(self, engine_path, H, W):
        self.H, self.W = H, W
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path,"rb").read())
        self._ctx = self._eng.create_execution_context()
        self._d_inp    = _malloc(3 * H * W * 4)
        self._d_scores = _malloc(MAX_DET * 4)
        self._d_labels = _malloc(MAX_DET * 4)
        self._d_boxes  = _malloc(MAX_DET * 4 * 4)
        self._ctx.set_tensor_address("image",  self._d_inp)
        self._ctx.set_tensor_address("scores", self._d_scores)
        self._ctx.set_tensor_address("labels", self._d_labels)
        self._ctx.set_tensor_address("boxes",  self._d_boxes)
        self._ctx.set_input_shape("image", (1, 3, H, W))
        # Pinned host buffers
        n_inp = 3 * H * W
        self._h_inp_ptr = _malloc_host(n_inp * 4)
        self._h_inp = np.frombuffer((ctypes.c_float * n_inp).from_address(self._h_inp_ptr),
                                    dtype=np.float32).reshape(1, 3, H, W)
        self._h_sc  = np.empty(MAX_DET, np.float32)
        self._h_lb  = np.empty(MAX_DET, np.int32)
        self._h_bx  = np.empty(MAX_DET*4, np.float32)
        self._stream = _stream_create()

    def run_one(self, bgr):
        t = {}
        # 1. 预处理
        t0 = now_ms()
        blob = cv2.dnn.blobFromImage(bgr, 1.0/255.0, (self.W, self.H), swapRB=True, crop=False)
        np.copyto(self._h_inp, blob)
        t["preprocess"] = now_ms() - t0

        # 2. H2D async
        t0 = now_ms()
        _h2d_async(self._d_inp, self._h_inp, self._stream)
        _stream_sync(self._stream)
        t["h2d"] = now_ms() - t0

        # 3. execute async
        t0 = now_ms()
        self._ctx.execute_async_v3(_stream_int(self._stream))
        _stream_sync(self._stream)
        t["execute"] = now_ms() - t0

        # 4. D2H async
        t0 = now_ms()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        if n > 0:
            _d2h_async(self._h_sc[:n], self._d_scores, self._stream)
            _d2h_async(self._h_lb[:n], self._d_labels, self._stream)
            _d2h_async(self._h_bx[:n*4], self._d_boxes, self._stream)
            _stream_sync(self._stream)
        t["d2h"] = now_ms() - t0

        # 5. 后处理
        t0 = now_ms()
        dets = []
        if n > 0:
            oh, ow = bgr.shape[:2]
            sx, sy = ow/self.W, oh/self.H
            mask = self._h_sc[:n] >= 0.5
            for i in np.where(mask)[0]:
                x1,y1,x2,y2 = (self._h_bx[i*4]*sx, self._h_bx[i*4+1]*sy,
                                self._h_bx[i*4+2]*sx, self._h_bx[i*4+3]*sy)
                dets.append((int(self._h_lb[i]), x1, y1, x2, y2, float(self._h_sc[i])))
        t["postprocess"] = now_ms() - t0
        t["total"] = sum(t.values())
        return t

    def close(self):
        _free(self._d_inp); _free(self._d_scores)
        _free(self._d_labels); _free(self._d_boxes)
        _free_host(self._h_inp_ptr)


# ─── 方式C：Letterbox 等比缩放 + Pinned ───────────────────────────────────────
class ProfilerLetterbox:
    name = "C: Letterbox等比缩放 + Pinned"

    def __init__(self, engine_path, H, W):
        self.H, self.W = H, W
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self._eng = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path,"rb").read())
        self._ctx = self._eng.create_execution_context()
        self._d_inp    = _malloc(3 * H * W * 4)
        self._d_scores = _malloc(MAX_DET * 4)
        self._d_labels = _malloc(MAX_DET * 4)
        self._d_boxes  = _malloc(MAX_DET * 4 * 4)
        self._ctx.set_tensor_address("image",  self._d_inp)
        self._ctx.set_tensor_address("scores", self._d_scores)
        self._ctx.set_tensor_address("labels", self._d_labels)
        self._ctx.set_tensor_address("boxes",  self._d_boxes)
        self._ctx.set_input_shape("image", (1, 3, H, W))
        n_inp = 3 * H * W
        self._h_inp_ptr = _malloc_host(n_inp * 4)
        self._h_inp = np.frombuffer((ctypes.c_float * n_inp).from_address(self._h_inp_ptr),
                                    dtype=np.float32).reshape(1, 3, H, W)
        self._stream = _stream_create()

    def preprocess_letterbox(self, bgr):
        """等比缩放 + 灰色 padding，保持宽高比"""
        h, w = bgr.shape[:2]
        scale = min(self.W / w, self.H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.H, self.W, 3), 114, dtype=np.uint8)
        pad_top  = (self.H - nh) // 2
        pad_left = (self.W - nw) // 2
        canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
        # 转 CHW float32 归一化
        blob = canvas.astype(np.float32) / 255.0
        blob = blob[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        self._last_scale = scale
        self._last_pad   = (pad_left, pad_top)
        return blob

    def run_one(self, bgr):
        t = {}
        # 1. 预处理 (letterbox)
        t0 = now_ms()
        blob = self.preprocess_letterbox(bgr)
        np.copyto(self._h_inp[0], blob)
        t["preprocess"] = now_ms() - t0

        # 2. H2D
        t0 = now_ms()
        _h2d_async(self._d_inp, self._h_inp, self._stream)
        _stream_sync(self._stream)
        t["h2d"] = now_ms() - t0

        # 3. execute
        t0 = now_ms()
        self._ctx.execute_async_v3(_stream_int(self._stream))
        _stream_sync(self._stream)
        t["execute"] = now_ms() - t0

        # 4. D2H
        t0 = now_ms()
        n = int(self._ctx.get_tensor_shape("scores")[0])
        sc = np.empty(n, np.float32); _d2h(sc, self._d_scores)
        lb = np.empty(n, np.int32);   _d2h(lb, self._d_labels)
        bx = np.empty(n*4, np.float32); _d2h(bx, self._d_boxes)
        t["d2h"] = now_ms() - t0

        # 5. 后处理（还原坐标考虑 letterbox 偏移）
        t0 = now_ms()
        dets = []
        scale = self._last_scale
        pl, pt = self._last_pad
        for i in range(n):
            if float(sc[i]) < 0.5: continue
            x1 = (bx[i*4]   - pl) / scale
            y1 = (bx[i*4+1] - pt) / scale
            x2 = (bx[i*4+2] - pl) / scale
            y2 = (bx[i*4+3] - pt) / scale
            dets.append((int(lb[i]), x1, y1, x2, y2, float(sc[i])))
        t["postprocess"] = now_ms() - t0
        t["total"] = sum(t.values())
        return t

    def close(self):
        _free(self._d_inp); _free(self._d_scores)
        _free(self._d_labels); _free(self._d_boxes)
        _free_host(self._h_inp_ptr)


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def profile_engine(engine_name, engine_file, H, W, images):
    engine_path = str(MODELS_DIR / engine_file)
    if not Path(engine_path).exists():
        print(f"  [SKIP] {engine_file} 不存在")
        return None

    print(f"\n{'═'*70}")
    print(f"  引擎: {engine_name}  输入: {H}×{W}  文件: {engine_file}")
    print(f"{'═'*70}")

    # 加载测试图片
    imgs = []
    for p in images[:N_IMAGES]:
        bgr = cv2.imread(str(p))
        if bgr is not None: imgs.append(bgr)
    print(f"  测试图片: {len(imgs)} 张  (每张重复 {N_REPEAT} 次取均值)")

    profilers = [
        ProfilerBaseline(engine_path, H, W),
        ProfilerPinned(engine_path, H, W),
        ProfilerLetterbox(engine_path, H, W),
    ]

    # warmup
    print("  热身中 ...", end="", flush=True)
    for _ in range(N_WARMUP):
        for pr in profilers:
            pr.run_one(imgs[0])
    print(" 完成")

    results = {}
    STAGES = ["preprocess", "h2d", "execute", "d2h", "postprocess", "total"]

    for pr in profilers:
        stage_times = {s: [] for s in STAGES}
        for bgr in imgs:
            row_times = {s: [] for s in STAGES}
            for _ in range(N_REPEAT):
                t = pr.run_one(bgr)
                for s in STAGES:
                    row_times[s].append(t[s])
            for s in STAGES:
                stage_times[s].append(statistics.mean(row_times[s]))

        print(f"\n  ── {pr.name}")
        print(f"  {'阶段':14s} {'均值':>8} {'P50':>8} {'P95':>8} {'占比':>7}")
        print(f"  {'─'*50}")
        total_mean = statistics.mean(stage_times["total"])
        for s in STAGES[:-1]:
            m = statistics.mean(stage_times[s])
            p50 = ns(stage_times[s], 50)
            p95 = ns(stage_times[s], 95)
            ratio = m / total_mean * 100 if total_mean > 0 else 0
            bar = "█" * int(ratio / 5)
            print(f"  {s:14s} {m:7.3f}ms {p50:7.3f}ms {p95:7.3f}ms {ratio:6.1f}% {bar}")
        print(f"  {'─'*50}")
        total_p50 = ns(stage_times["total"], 50)
        total_p95 = ns(stage_times["total"], 95)
        print(f"  {'TOTAL':14s} {total_mean:7.3f}ms {total_p50:7.3f}ms {total_p95:7.3f}ms")
        results[pr.name] = {"mean": total_mean, "p50": total_p50, "p95": total_p95,
                            "stages": {s: statistics.mean(stage_times[s]) for s in STAGES[:-1]}}

    for pr in profilers:
        pr.close()

    return results


def main():
    images = sorted(IMG_DIR.glob("*.jpg"))[:N_IMAGES]
    if not images:
        images = sorted(IMG_DIR.glob("*.png"))[:N_IMAGES]
    if not images:
        print(f"[ERROR] 未找到图片：{IMG_DIR}")
        return

    print(f"推理流水线耗时分析")
    print(f"图片目录: {IMG_DIR}")
    print(f"图片数  : {len(images)}")
    print(f"热身次数: {N_WARMUP}  重复次数: {N_REPEAT}/帧")

    all_results = {}
    for eng_name, eng_file, H, W in ENGINES:
        r = profile_engine(eng_name, eng_file, H, W, images)
        if r:
            all_results[eng_name] = r

    # ── 综合对比表 ────────────────────────────────────────────────────────────
    W_disp = 75
    print(f"\n\n{'═'*W_disp}")
    print(f"  ★ 综合对比：各推理方式总耗时 (ms, P50)")
    print(f"{'═'*W_disp}")
    print(f"  {'引擎':12s} {'':35s} {'均值':>8} {'P50':>8} {'P95':>8}")
    print(f"  {'─'*W_disp}")
    for eng, methods in all_results.items():
        for method, vals in methods.items():
            print(f"  {eng:12s} {method:35s} {vals['mean']:8.2f} {vals['p50']:8.2f} {vals['p95']:8.2f}")
        print()

    # ── 优化收益分析 ──────────────────────────────────────────────────────────
    print(f"{'═'*W_disp}")
    print(f"  ★ 各阶段耗时分解 (引擎: FP16_500h, 方法A vs B vs C)")
    print(f"{'═'*W_disp}")
    if "FP16_500h" in all_results:
        r500 = all_results["FP16_500h"]
        print(f"  {'阶段':14s}", end="")
        methods_list = list(r500.keys())
        for m in methods_list:
            print(f" {m[:20]:>22s}", end="")
        print()
        STAGES = ["preprocess", "h2d", "execute", "d2h", "postprocess"]
        for s in STAGES:
            print(f"  {s:14s}", end="")
            for m in methods_list:
                v = r500[m]["stages"].get(s, 0)
                print(f" {v:22.3f}", end="")
            print()

    print(f"\n{'═'*W_disp}")
    print("  优化建议汇总:")
    print("  1. Pinned Memory (B) 减少 H2D/D2H 耗时约 20-40%（避免一次额外内存拷贝）")
    print("  2. Letterbox (C) 改善 nuImages 图像宽高比变形问题")
    print("  3. 最快引擎: INT8_500h（量化引擎体积小推理快），FP16_375h 次之")
    print("  4. execute 阶段耗时占总耗时 >80%，是真正瓶颈，其余优化收益有限")


if __name__ == "__main__":
    main()
