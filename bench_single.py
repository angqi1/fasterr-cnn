#!/usr/bin/env python3
"""单引擎独立基准测试 - 每次只加载一个引擎，避免多引擎互干扰。
用法: python3 bench_single.py <engine_name>
  engine_name: fp16_375 | int8_375 | fp16_500 | int8_500 | fp16_700
"""
import sys, cv2, numpy as np, ctypes, glob, time, tensorrt as trt

MDIR = "install/faster_rcnn_ros/share/faster_rcnn_ros/models/"
ENGINES = {
    "fp16_375":  ("faster_rcnn_375.engine",      375, 1242, "FP16_375h"),
    "int8_375":  ("faster_rcnn_375_int8.engine",  375, 1242, "INT8_375h"),
    "fp16_500":  ("faster_rcnn_500.engine",        500, 1242, "FP16_500h"),
    "int8_500":  ("faster_rcnn_500_int8.engine",   500, 1242, "INT8_500h"),
    "fp16_700":  ("faster_rcnn_700.engine",        700, 1242, "FP16_700h"),
}

key = sys.argv[1] if len(sys.argv) > 1 else "fp16_375"
fname, H, W, label = ENGINES[key]
path = MDIR + fname

IMGS = sorted(glob.glob("test_images/kitti_100/images/*.png"))[:20]
assert IMGS, "找不到测试图片"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_DET = 2000

log = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(log, "")
cu = ctypes.CDLL("libcudart.so")

print(f"加载引擎: {path}")
with open(path, "rb") as f:
    eng = trt.Runtime(log).deserialize_cuda_engine(f.read())
ctx = eng.create_execution_context()

# 分配 GPU 显存
d_i = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_i), 3*H*W*4)
d_s = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_s), MAX_DET*4)
d_l = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_l), MAX_DET*4)
d_b = ctypes.c_void_p(); cu.cudaMalloc(ctypes.byref(d_b), MAX_DET*16)

ctx.set_tensor_address("image",  int(d_i.value))
ctx.set_tensor_address("scores", int(d_s.value))
ctx.set_tensor_address("labels", int(d_l.value))
ctx.set_tensor_address("boxes",  int(d_b.value))

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

# 预热
print(f"预热 10 帧...")
for p in (IMGS * 4)[:10]:
    infer(cv2.imread(p))

# 基准测试
N_REP = 5
N_PER = len(IMGS[:20])
print(f"基准 {N_REP} 轮 × {N_PER} 张图 = {N_REP*N_PER} 次推理...")
ts = []
for _ in range(N_REP):
    for p in IMGS[:N_PER]:
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
