#!/usr/bin/env python3
"""
使用真实校准数据构建 INT8 TensorRT 引擎。

支持多分辨率：375x1242 / 500x1242 / 700x1242

INT8 校准流程（Post-Training Quantization, PTQ）：
  1. 加载校准集图像（KITTI 100 张）
  2. 对每张图预处理为模型输入格式
  3. TRT EntropyCalibrator2 运行校准，测量各激活层的动态范围
  4. 生成校准缓存文件（.cache），保存量化比例因子
  5. 用校准缓存构建真正的 INT8 引擎

用法:
  python3 build_int8_calibrated_engine.py                 # 默认 375x1242
  python3 build_int8_calibrated_engine.py --height 500   # 500x1242
  python3 build_int8_calibrated_engine.py --height 375 500 700  # 批量构建

输出:
  faster_rcnn_375_int8_calib.engine   (~58MB，真正INT8，约 40-55ms GPU)
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import tensorrt as trt
import ctypes

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(WORKSPACE_DIR, 'src/faster_rcnn_ros/models')
INSTALL_DIR   = os.path.join(WORKSPACE_DIR, 'install/faster_rcnn_ros/share/faster_rcnn_ros/models')
CALIB_IMAGES  = sorted(glob.glob(os.path.join(WORKSPACE_DIR, 'test_images/kitti_100/images/*.png')))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ImageNet mean/std (与训练预处理一致)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 预处理函数（与推理节点保持一致）
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """BGR uint8 → CHW float32 normalized blob, shape (1,3,h,w)"""
    img = cv2.resize(img_bgr, (w, h))
    img = img[:, :, ::-1].astype(np.float32) / 255.0       # BGR→RGB, /255
    img = (img - MEAN) / STD                                # normalize
    img = img.transpose(2, 0, 1)[np.newaxis]                # HWC→CHW→NCHW
    return np.ascontiguousarray(img, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# INT8 Entropy Calibrator
# ─────────────────────────────────────────────────────────────────────────────
class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    """
    使用真实图像进行 INT8 后训练量化校准。
    trt.IInt8EntropyCalibrator2 是 TensorRT 推荐的默认校准方法。
    """
    def __init__(self, images: list, h: int, w: int, cache_file: str,
                 batch_size: int = 1):
        super().__init__()
        self.h = h
        self.w = w
        self.images = images
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0

        # 在设备上分配输入 buffer
        self._buf_size = batch_size * 3 * h * w * 4  # float32 bytes
        _cudart = ctypes.cdll.LoadLibrary('libcudart.so')
        # 设置函数签名避免类型错误
        _cudart.cudaMalloc.restype  = ctypes.c_int
        _cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        _cudart.cudaMemcpy.restype  = ctypes.c_int
        _cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int]
        _cudart.cudaFree.restype    = ctypes.c_int
        _cudart.cudaFree.argtypes   = [ctypes.c_void_p]
        self._d_input = ctypes.c_void_p()
        ret = _cudart.cudaMalloc(ctypes.byref(self._d_input), self._buf_size)
        if ret != 0:
            raise RuntimeError(f"cudaMalloc failed (code {ret}), 显存不足?")
        self._cudart = _cudart
        print(f"  [Calibrator] 校准集: {len(images)} 张图像, "
              f"输入: {h}x{w}, 缓存: {cache_file}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.images):
            return None
        batch_imgs = []
        for i in range(self.batch_size):
            idx = self.current_index + i
            img = cv2.imread(self.images[idx])
            if img is None:
                print(f"  警告: 无法读取图像 {self.images[idx]}")
                continue
            blob = preprocess(img, self.h, self.w)
            batch_imgs.append(blob)
        if not batch_imgs:
            return None
        batch = np.concatenate(batch_imgs, axis=0)
        # H2D copy
        self._cudart.cudaMemcpy(
            self._d_input,
            batch.ctypes.data_as(ctypes.c_void_p),
            batch.nbytes,
            1  # cudaMemcpyHostToDevice
        )
        self.current_index += self.batch_size
        progress = self.current_index / len(self.images) * 100
        print(f"\r  校准进度: {self.current_index}/{len(self.images)} ({progress:.0f}%)", end='', flush=True)
        return [int(self._d_input.value)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"  [Calibrator] 读取已有校准缓存: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"\n  [Calibrator] 校准缓存已写入: {self.cache_file} ({len(cache)} bytes)")

    def __del__(self):
        if hasattr(self, '_cudart') and hasattr(self, '_d_input'):
            try:
                self._cudart.cudaFree(self._d_input)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 引擎构建函数
# ─────────────────────────────────────────────────────────────────────────────
def build_int8_engine(onnx_path: str, engine_path: str, h: int, w: int,
                       calib_images: list, cache_file: str,
                       workspace_gb: int = 4) -> bool:
    """
    用真实校准数据构建 INT8 TRT 引擎。
    同时启用 FP16 作为 INT8 不支持层的 fallback。
    """
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = workspace_gb * (1 << 30)

        # 启用 FP16（INT8 不支持的层 fallback 到 FP16，不是 FP32）
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  [Build] FP16 fallback 已启用")

        # 启用 INT8
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  [Build] INT8 已启用")
        else:
            print("  [警告] 此平台不支持快速 INT8，跳过")
            return False

        # 设置校准器
        calibrator = Int8EntropyCalibrator2(
            calib_images, h, w, cache_file, batch_size=1
        )
        config.int8_calibrator = calibrator

        # 解析 ONNX
        print(f"  [Build] 解析 ONNX: {onnx_path}")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("  [错误] ONNX 解析失败:")
                for i in range(parser.num_errors):
                    print(f"    {parser.get_error(i)}")
                return False
        print(f"  [Build] ONNX 解析成功，网络层数: {network.num_layers}")

        # 设置固定输入形状
        input_shape = (1, 3, h, w)
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name,
                          min=input_shape, opt=input_shape, max=input_shape)
        config.add_optimization_profile(profile)
        print(f"  [Build] 输入形状: {input_shape}")

        # 构建引擎（这步会触发校准）
        print(f"\n  [Build] 开始构建引擎（含 INT8 校准，约 15-20 分钟）...")
        serialized = builder.build_serialized_network(network, config)
        print()  # 换行

        if serialized is None:
            print("  [错误] 引擎构建失败")
            return False

        with open(engine_path, 'wb') as f:
            f.write(serialized)

        size_mb = os.path.getsize(engine_path) / 1024 / 1024
        print(f"  ✅ 引擎已保存: {engine_path} ({size_mb:.1f} MB)")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 验证引擎精度（构建后检查实际 INT8 层占比）
# ─────────────────────────────────────────────────────────────────────────────
def verify_engine_precision(engine_path: str):
    """检查引擎 binding 精度和 inspector 信息"""
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    print(f"\n  [验证] 引擎: {os.path.basename(engine_path)}")
    print(f"  [验证] Binding 数: {engine.num_bindings}")
    for i in range(engine.num_bindings):
        name  = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)
        print(f"    [{i}] {name}: {dtype}, shape={shape}")

    # Inspector 检查
    insp = engine.create_engine_inspector()
    raw = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    if isinstance(raw, str):
        int8_n = raw.count('"INT8"')
        fp16_n = raw.count('"FP16"') + raw.count('"Half"')
        fp32_n = raw.count('"Float"') + raw.count('"FP32"')
        total  = int8_n + fp16_n + fp32_n
        if total > 0:
            print(f"  [验证] 层精度统计 (inspector): "
                  f"INT8={int8_n} ({int8_n/total*100:.0f}%), "
                  f"FP16={fp16_n} ({fp16_n/total*100:.0f}%), "
                  f"FP32={fp32_n} ({fp32_n/total*100:.0f}%)")
            if int8_n > 0:
                print(f"  ✅ 确认: 真正的 INT8 层 已存在")
            else:
                print(f"  ⚠️  Inspector 未找到 INT8 标记（可能是 TRT 格式问题）")
        else:
            print(f"  [验证] Inspector 精度解析: 未找到精度信息")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='构建带真实校准数据的INT8 TensorRT引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--height', type=int, nargs='+', default=[375],
        metavar='H',
        help='输入高度，可多个: --height 375 500 700 (默认: 375)'
    )
    parser.add_argument(
        '--width', type=int, default=1242,
        help='输入宽度 (默认: 1242)'
    )
    parser.add_argument(
        '--calib-dir', default=os.path.join(WORKSPACE_DIR, 'test_images/kitti_100/images'),
        help='校准图像目录'
    )
    parser.add_argument(
        '--workspace', type=int, default=4,
        help='TRT workspace 大小 (GB, 默认: 4)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='即使引擎已存在也重新构建'
    )
    args = parser.parse_args()

    # 加载校准图像
    calib_dir = args.calib_dir
    calib_images = sorted(glob.glob(os.path.join(calib_dir, '*.png')))
    if not calib_images:
        calib_images = sorted(glob.glob(os.path.join(calib_dir, '*.jpg')))
    if not calib_images:
        print(f"❌ 校准目录中没有图像: {calib_dir}")
        sys.exit(1)
    print(f"校准图像: {len(calib_images)} 张 (来自 {calib_dir})")

    W = args.width
    results = []

    for H in args.height:
        print(f"\n{'='*60}")
        print(f" 构建 INT8 引擎: {H}x{W}")
        print(f"{'='*60}")

        # 输入文件
        onnx_name = f'fasterrcnn_nuscenes_fixed_{H}x{W}.onnx'
        onnx_path = os.path.join(MODEL_DIR, onnx_name)
        if not os.path.exists(onnx_path):
            # 也检查 install 目录
            onnx_path2 = os.path.join(INSTALL_DIR, onnx_name)
            if os.path.exists(onnx_path2):
                onnx_path = onnx_path2
            else:
                print(f"  ❌ 找不到 ONNX: {onnx_path}")
                results.append((H, W, False, "ONNX not found"))
                continue

        # 输出文件（同时写入 src 和 install 目录）
        engine_name = f'faster_rcnn_{H}_int8_calib.engine'
        engine_path = os.path.join(MODEL_DIR, engine_name)
        engine_install = os.path.join(INSTALL_DIR, engine_name)
        cache_file = os.path.join(MODEL_DIR, f'int8_calib_{H}x{W}.cache')

        print(f"  ONNX:   {onnx_path}")
        print(f"  Engine: {engine_path}")
        print(f"  Cache:  {cache_file}")

        if os.path.exists(engine_path) and not args.force:
            size_mb = os.path.getsize(engine_path) / 1024 / 1024
            print(f"  [SKIP] 引擎已存在 ({size_mb:.1f} MB)，使用 --force 重新构建")
            verify_engine_precision(engine_path)
            results.append((H, W, True, f"already exists ({size_mb:.1f}MB)"))
            continue

        success = build_int8_engine(
            onnx_path, engine_path, H, W,
            calib_images, cache_file,
            workspace_gb=args.workspace
        )

        if success:
            # 同步到 install 目录
            os.makedirs(os.path.dirname(engine_install), exist_ok=True)
            import shutil
            shutil.copy2(engine_path, engine_install)
            print(f"  已同步到: {engine_install}")
            verify_engine_precision(engine_path)
            results.append((H, W, True, "built successfully"))
        else:
            results.append((H, W, False, "build failed"))

    # 汇总
    print(f"\n{'='*60}")
    print(" 构建结果汇总")
    print(f"{'='*60}")
    for H, W, ok, msg in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {H}x{W}: {msg}")


if __name__ == '__main__':
    main()
