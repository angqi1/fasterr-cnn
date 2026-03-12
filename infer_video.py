#!/usr/bin/env python3
"""
Faster R-CNN TensorRT 视频推理 + 标注脚本
输入 : fixed_4k_video.mp4  (3840x2160, 30fps)
引擎 : faster_rcnn.engine  (480x640, FP16)
输出 : annotated_output.mp4

TRT 8.5 DDS (数据相关形状) 正确推理方式:
  execute_async_v3 + IOutputAllocator
"""

import cv2
import numpy as np
import tensorrt as trt
import ctypes
import time
import os

# ctypes cudart 封装
_cudart = ctypes.CDLL('libcudart.so.11.0', mode=ctypes.RTLD_GLOBAL)

def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    assert _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size)) == 0
    return ptr.value

def cuda_free(ptr):
    _cudart.cudaFree(ctypes.c_void_p(ptr))

def cuda_h2d(dst, src_np):
    assert _cudart.cudaMemcpy(ctypes.c_void_p(dst),
        src_np.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src_np.nbytes), ctypes.c_int(1)) == 0

def cuda_d2h(dst_np, src):
    assert _cudart.cudaMemcpy(
        dst_np.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(src),
        ctypes.c_size_t(dst_np.nbytes), ctypes.c_int(2)) == 0

def cuda_stream_create():
    s = ctypes.c_void_p()
    assert _cudart.cudaStreamCreate(ctypes.byref(s)) == 0
    return s.value

def cuda_stream_sync(s):
    _cudart.cudaStreamSynchronize(ctypes.c_void_p(s))


class FixedOutputAllocator(trt.IOutputAllocator):
    """TRT 8.5 IOutputAllocator: 预分配大缓冲区，推理后获取实际尺寸"""
    def __init__(self, max_bytes):
        super().__init__()
        self._max_bytes = max_bytes
        self._ptrs   = {}   # name -> (ptr, capacity)
        self._shapes = {}   # name -> shape tuple

    def reallocate_output(self, name, memory, size, alignment):
        existing = self._ptrs.get(name)
        if existing and existing[1] >= size:
            return existing[0]
        alloc = max(size, self._max_bytes)
        ptr = cuda_malloc(alloc)
        self._ptrs[name] = (ptr, alloc)
        return ptr

    def notify_shape(self, name, shape):
        self._shapes[name] = tuple(shape)

    def ptr(self, name):
        return self._ptrs.get(name, (None, 0))[0]

    def shape(self, name):
        return self._shapes.get(name, (0,))


# 配置
ENGINE_PATH = '/home/nvidia/ros2_ws/faster_rcnn.engine'
LABELS_PATH = '/home/nvidia/ros2_ws/src/faster_rcnn_ros/models/labels.txt'
VIDEO_IN    = '/home/nvidia/ros2_ws/fixed_1080p_video.mp4'
VIDEO_OUT   = '/home/nvidia/ros2_ws/annotated_1080p_output.mp4'
MAX_FRAMES  = 100   # 仅处理前N帧，设为None则处理全部
THRESHOLD   = 0.5
MODEL_H     = 480
MODEL_W     = 640
MAX_BYTES   = 4000 * 4 * 4   # 4000 检测框 x 4bytes x 4 (boxes每个4float)

with open(LABELS_PATH) as f:
    LABELS = [l.strip() for l in f]
np.random.seed(42)
COLORS = np.random.randint(80, 230, size=(len(LABELS), 3), dtype=np.uint8).tolist()

# 加载引擎
print("正在加载 TRT 引擎...")
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')
logger  = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(ENGINE_PATH, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
print("引擎加载完成")

# 输入缓冲区
h_input = np.zeros((1, 3, MODEL_H, MODEL_W), dtype=np.float32)
d_input = cuda_malloc(h_input.nbytes)
context.set_input_shape("input_image", (1, 3, MODEL_H, MODEL_W))
context.set_tensor_address("input_image", d_input)

# 输出 allocator
allocator = FixedOutputAllocator(MAX_BYTES)
for name in ("scores", "labels", "boxes"):
    context.set_output_allocator(name, allocator)

stream = cuda_stream_create()


def infer(bgr_frame):
    resized = cv2.resize(bgr_frame, (MODEL_W, MODEL_H))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    np.copyto(h_input,
              rgb.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0)
    cuda_h2d(d_input, h_input)
    context.execute_async_v3(stream)
    cuda_stream_sync(stream)

    n = allocator.shape("scores")[0] if allocator.shape("scores") else 0
    if n == 0:
        return (np.empty(0, np.float32),
                np.empty(0, np.int32),
                np.empty((0, 4), np.float32))
    h_sc = np.empty(n,      dtype=np.float32)
    h_lb = np.empty(n,      dtype=np.int32)
    h_bx = np.empty((n, 4), dtype=np.float32)
    cuda_d2h(h_sc, allocator.ptr("scores"))
    cuda_d2h(h_lb, allocator.ptr("labels"))
    cuda_d2h(h_bx, allocator.ptr("boxes"))
    return h_sc, h_lb, h_bx


# 预热
print("引擎预热...")
dummy_bgr = np.zeros((MODEL_H, MODEL_W, 3), dtype=np.uint8)
infer(dummy_bgr)
print("预热完成\n")

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.4
THICKNESS  = 3


def draw_detections(frame, scores, labels, boxes, orig_h, orig_w):
    sx, sy = orig_w / MODEL_W, orig_h / MODEL_H
    count = 0
    for sc, lb, bx in zip(scores, labels, boxes):
        if sc < THRESHOLD:
            continue
        x1 = int(np.clip(bx[0] * sx, 0, orig_w - 1))
        y1 = int(np.clip(bx[1] * sy, 0, orig_h - 1))
        x2 = int(np.clip(bx[2] * sx, 0, orig_w - 1))
        y2 = int(np.clip(bx[3] * sy, 0, orig_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        lb    = int(lb)
        color = tuple(COLORS[lb % len(COLORS)])
        cls   = LABELS[lb] if 0 <= lb < len(LABELS) else f'cls{lb}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS + 1)
        text = f'{cls} {sc:.2f}'
        (tw, th), bl = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        ty = max(y1 - 6, th + 8)
        cv2.rectangle(frame, (x1, ty - th - bl - 4),
                      (x1 + tw + 8, ty + bl), color, -1)
        cv2.putText(frame, text, (x1 + 4, ty - 2),
                    FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)
        count += 1
    return count


# 视频循环
cap     = cv2.VideoCapture(VIDEO_IN)
fps     = cap.get(cv2.CAP_PROP_FPS)
orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"输入 : {VIDEO_IN}")
print(f"  {orig_w}x{orig_h}  {fps:.2f}fps  {total_f}帧")
print(f"输出 : {VIDEO_OUT}")
print(f"阈值 : {THRESHOLD}\n")

# 输出保持1080p分辨率
OUT_W, OUT_H = 1920, 1080
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (OUT_W, OUT_H))

t0         = time.time()
frame_no   = 0
total_dets = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1
    if MAX_FRAMES is not None and frame_no > MAX_FRAMES:
        break
    scores, labels, boxes = infer(frame)
    n_dets      = draw_detections(frame, scores, labels, boxes, orig_h, orig_w)
    total_dets += n_dets

    elapsed  = time.time() - t0
    fps_cur  = frame_no / elapsed if elapsed > 0 else 0
    remain_s = (total_f - frame_no) / fps_cur if fps_cur > 0 else 0
    hud = (f'Frame {frame_no}/{total_f}  Dets:{n_dets}  '
           f'{fps_cur:.1f}fps  ETA:{remain_s:.0f}s')
    # 缩放到1080p输出（避免4K软编码瓶颈，速度提升约3x）
    out_frame = cv2.resize(frame, (OUT_W, OUT_H))
    cv2.putText(out_frame, hud, (20, 50), FONT, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(out_frame, hud, (20, 50), FONT, 1.2, (0, 0, 0),   1, cv2.LINE_AA)
    writer.write(out_frame)
    if frame_no % 100 == 0 or frame_no == 1:
        print(f"  [{frame_no:5d}/{total_f}]  检测:{n_dets:3d}  "
              f"{fps_cur:5.1f}fps  ETA:{remain_s:.0f}s")

cap.release()
writer.release()
cuda_free(d_input)

elapsed = time.time() - t0
print(f"\n完成!")
print(f"  总帧数  : {frame_no}")
print(f"  耗时    : {elapsed:.1f}s  平均 {frame_no/elapsed:.1f}fps")
print(f"  平均检测: {total_dets/max(frame_no,1):.1f} 个/帧")
fsize = os.path.getsize(VIDEO_OUT)/1024/1024
print(f"  输出文件: {VIDEO_OUT}  ({fsize:.1f} MB)")
