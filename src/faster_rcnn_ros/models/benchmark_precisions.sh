#!/bin/bash
# ============================================================================
# 性能基准测试脚本：对比 FP32 / FP16 / INT8 引擎的延迟和吞吐量
# 同时在 KITTI 图片上运行实际推理，输出检测数量对比
#
# 用法: bash benchmark_precisions.sh [--height 500] [--gpu-max]
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
HEIGHT=${1:-500}
WIDTH=1242
GPU_MAX=${2:-""}  # 传 --gpu-max 则先锁定 GPU 最大频率

echo "============================================================"
echo " Faster R-CNN TensorRT 性能基准测试"
echo " 平台: Jetson AGX Orin | TRT 8.5.2 | JetPack 5.1.2"
echo " 分辨率: ${HEIGHT}×${WIDTH}"
echo "============================================================"

# ── 检查 GPU 频率 ──────────────────────────────────────────────────────────
GPU_CUR=$(cat /sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/cur_freq 2>/dev/null || echo "unknown")
GPU_MAX_FREQ=$(cat /sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/max_freq 2>/dev/null || echo "unknown")

echo ""
echo "[GPU 频率] 当前: ${GPU_CUR} Hz | 最大: ${GPU_MAX_FREQ} Hz"

if [ "$GPU_CUR" != "$GPU_MAX_FREQ" ]; then
    echo "[WARNING] GPU 未运行在最高频率！推理时间会偏高"
    echo "  建议先运行: sudo jetson_clocks"
    echo ""
fi

SHAPES="image:1x3x${HEIGHT}x${WIDTH}"

# ── trtexec 基准测试函数 ─────────────────────────────────────────────────
benchmark_engine() {
    local engine_file="$1"
    local precision_name="$2"

    if [ ! -f "$engine_file" ]; then
        echo "[$precision_name] 引擎不存在: $engine_file — 跳过"
        return
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo "  测试: $precision_name"
    echo "  引擎: $(basename "$engine_file") ($(du -h "$engine_file" | cut -f1))"
    echo "────────────────────────────────────────"

    # 使用 trtexec 的内置基准测试（100次迭代，10次预热）
    $TRTEXEC \
        --loadEngine="$engine_file" \
        --shapes="$SHAPES" \
        --iterations=100 \
        --warmUp=3000 \
        --avgRuns=10 \
        --useSpinWait 2>&1 | grep -E "mean|median|percentile|Throughput|GPU Compute Time|Host Latency|Enqueue"
}

# ── 逐精度测试 ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  trtexec 纯推理基准（排除 I/O、前后处理）"
echo "============================================================"

benchmark_engine "${SCRIPT_DIR}/faster_rcnn_${HEIGHT}_fp32.engine" "FP32"
benchmark_engine "${SCRIPT_DIR}/faster_rcnn_${HEIGHT}_fp16.engine" "FP16"
benchmark_engine "${SCRIPT_DIR}/faster_rcnn_${HEIGHT}.engine"      "FP16(原始)"
benchmark_engine "${SCRIPT_DIR}/faster_rcnn_${HEIGHT}_int8.engine" "INT8"

echo ""
echo "============================================================"
echo "  基准测试完成"
echo "============================================================"
echo ""
echo "提示："
echo "  - GPU Compute Time mean = 纯 GPU 推理耗时（不含前后处理）"
echo "  - Host Latency mean = 端到端延迟（含 H2D/D2H memcpy）"
echo "  - Throughput = 每秒可处理的推理次数"
echo ""
echo "  若 GPU Compute 远高于预期，请运行："
echo "    sudo jetson_clocks         # 锁定 GPU/CPU 最大频率"
echo "    sudo nvpmodel -m 0         # 确保 MAXN 功率模式"
