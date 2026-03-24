#!/usr/bin/env python3
"""单引擎独立基准测试 - 每次只加载一个引擎，避免多引擎互干扰。
用法: python3 bench_single.py <engine_name>
  engine_name:
    fp16_375            FP16 375×1242  (基准)
    int8_375            INT8(无校准) 375×1242
    fp16_500            FP16 500×1242
    int8_500            INT8(fallback) 500×1242
    fp16_700            FP16 700×1242
    fp16_320            FP16 320×960   (非等比缩放，宽高比失真9.4%)
    cublaslt_375        FP16+cuBLASLt 375×1242
    fp16_320x1060       FP16 320×1060  (等比缩放，宽高比误差0.015%)
    all                 依次测试所有可用引擎并打印对比表
"""
import sys, cv2, numpy as np, ctypes, glob, time, tensorrt as trt, os

MDIR = "install/faster_rcnn_ros/share/faster_rcnn_ros/models/"
ENGINES = {
    "fp16_375":      ("faster_rcnn_375.engine",           375, 1242, "FP16_375h"),
    "int8_375":      ("faster_rcnn_375_int8.engine",       375, 1242, "INT8_375h(无校准)"),
    "fp16_500":      ("faster_rcnn_500.engine",            500, 1242, "FP16_500h"),
    "int8_500":      ("faster_rcnn_500_int8.engine",       500, 1242, "INT8_500h(fallback)"),
    "fp16_700":      ("faster_rcnn_700.engine",            700, 1242, "FP16_700h"),
    # 新引擎：低分辨率 + cuBLASLt 对比
    "fp16_320":      ("faster_rcnn_320.engine",            320,  960, "FP16_320×960(非等比)"),
    "cublaslt_375":  ("faster_rcnn_375_cublaslt.engine",   375, 1242, "FP16_375h+cuBLASLt"),
    "fp16_320x1060": ("faster_rcnn_320x1060.engine",       320, 1060, "FP16_320×1060(等比)"),
}

key = sys.argv[1] if len(sys.argv) > 1 else "fp16_375"

IMGS = sorted(glob.glob("test_images/kitti_100/images/*.png"))[:20]
assert IMGS, "找不到测试图片"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_DET = 2000

log = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(log, "")
cu = ctypes.CDLL("libcudart.so")


def run_bench(fname, H, W, label, n_rep=5):
    path = MDIR + fname
    if not os.path.exists(path):
        print(f"[跳过] 引擎不存在: {path}")
        return None

    print(f"\n加载引擎: {path}")
    with open(path, "rb") as f:
        eng = trt.Runtime(log).deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()

    d_i = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_i), 3*H*W*4)
    d_s = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_s), MAX_DET*4)
    d_l = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_l), MAX_DET*4)
    d_b = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_b), MAX_DET*16)

    ctx.set_tensor_address("image",  int(d_i.value))
    ctx.set_tensor_address("scores", int(d_s.value))
    ctx.set_tensor_address("labels", int(d_l.value))
    ctx.set_tensor_address("boxes",  int(d_b.value))
    ctx.set_input_shape("image", (1, 3, H, W))

    def infer(img):
        r = cv2.resize(img, (W, H))
        b = (r[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        b = np.ascontiguousarray(b.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
        cu.cudaMemcpy(d_i, b.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_size_t(b.nbytes), ctypes.c_int(1))
        cu.cudaDeviceSynchronize()
        t0 = time.perf_counter()
        ctx.execute_async_v3(0)
        cu.cudaDeviceSynchronize()
        return (time.perf_counter() - t0) * 1000

    print(f"预热 10 帧...")
    for p in (IMGS * 4)[:10]:
        infer(cv2.imread(p))

    n_per = len(IMGS[:20])
    print(f"基准 {n_rep} 轮 × {n_per} 张图 = {n_rep*n_per} 次推理...")
    ts = []
    for _ in range(n_rep):
        for p in IMGS[:n_per]:
            ts.append(infer(cv2.imread(p)))

    t = np.array(ts)
    print(f"\n{'='*50}")
    print(f"引擎: {label}  ({fname})")
    print(f"  次数   : {len(t)}")
    print(f"  均值   : {t.mean():.1f} ms")
    print(f"  中位数 : {np.median(t):.1f} ms")
    print(f"  最小值 : {t.min():.1f} ms")
    print(f"  最大值 : {t.max():.1f} ms")
    print(f"  P90    : {np.percentile(t, 90):.1f} ms")
    print(f"{'='*50}")

    # 释放 GPU 内存
    for d in [d_i, d_s, d_l, d_b]:
        cu.cudaFree(d)
    del ctx, eng

    return {"label": label, "fname": fname, "median": float(np.median(t)),
            "mean": float(t.mean()), "min": float(t.min()), "p90": float(np.percentile(t, 90))}


if key == "all":
    results = []
    for k, (fname, H, W, label) in ENGINES.items():
        r = run_bench(fname, H, W, label)
        if r:
            results.append(r)
    if results:
        print("\n\n" + "="*70)
        print(f"{'引擎':<28} {'中位数':>8} {'均值':>8} {'最小':>8} {'P90':>8}")
        print("-"*70)
        baseline = results[0]["median"]
        for r in results:
            speedup = f"{baseline/r['median']:.2f}x"
            print(f"{r['label']:<28} {r['median']:>7.1f}ms {r['mean']:>7.1f}ms "
                  f"{r['min']:>7.1f}ms {r['p90']:>7.1f}ms  ({speedup})")
        print("="*70)
else:
    if key not in ENGINES:
        print(f"未知引擎: {key}\n可用: {', '.join(ENGINES.keys())}")
        sys.exit(1)
    fname, H, W, label = ENGINES[key]
    run_bench(fname, H, W, label)
