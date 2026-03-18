#!/bin/bash
# ============================================================================
# 构建 FP32 / FP16 / INT8 三种精度的 TensorRT 引擎
# 用法: bash build_all_precisions.sh [--height 500] [--width 1242]
#
# 需要已有修复好的 ONNX 文件（由 build_engine.py 步骤 1-9 生成）
# 若 ONNX 不存在，会先运行 build_engine.py 修复 ONNX（但跳过引擎构建）
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

# 默认分辨率
HEIGHT=${1:-500}
WIDTH=${2:-1242}

FIXED_ONNX="${SCRIPT_DIR}/fasterrcnn_nuscenes_fixed_${HEIGHT}x${WIDTH}.onnx"

# 检查修复后的 ONNX 是否存在
if [ ! -f "$FIXED_ONNX" ]; then
    echo "[INFO] 修复后的 ONNX 不存在，先运行 build_engine.py 生成..."
    cd "$SCRIPT_DIR"
    python3 build_engine.py --height "$HEIGHT" --width "$WIDTH"
fi

echo "========================================"
echo " 使用 ONNX: $FIXED_ONNX"
echo " 分辨率:    ${HEIGHT}x${WIDTH}"
echo "========================================"

SHAPES="image:1x3x${HEIGHT}x${WIDTH}"

# ── FP32 ──────────────────────────────────────────────────────────────────
ENGINE_FP32="${SCRIPT_DIR}/faster_rcnn_${HEIGHT}_fp32.engine"
if [ -f "$ENGINE_FP32" ]; then
    echo "[SKIP] FP32 引擎已存在: $ENGINE_FP32"
else
    echo ""
    echo "[1/3] 构建 FP32 引擎..."
    $TRTEXEC \
        --onnx="$FIXED_ONNX" \
        --saveEngine="$ENGINE_FP32" \
        --minShapes="$SHAPES" \
        --optShapes="$SHAPES" \
        --maxShapes="$SHAPES" \
        --workspace=4096
    echo "[OK] FP32 引擎: $ENGINE_FP32 ($(du -h "$ENGINE_FP32" | cut -f1))"
fi

# ── FP16 ──────────────────────────────────────────────────────────────────
ENGINE_FP16="${SCRIPT_DIR}/faster_rcnn_${HEIGHT}_fp16.engine"
if [ -f "$ENGINE_FP16" ]; then
    echo "[SKIP] FP16 引擎已存在: $ENGINE_FP16"
else
    echo ""
    echo "[2/3] 构建 FP16 引擎..."
    $TRTEXEC \
        --onnx="$FIXED_ONNX" \
        --saveEngine="$ENGINE_FP16" \
        --fp16 \
        --minShapes="$SHAPES" \
        --optShapes="$SHAPES" \
        --maxShapes="$SHAPES" \
        --workspace=4096
    echo "[OK] FP16 引擎: $ENGINE_FP16 ($(du -h "$ENGINE_FP16" | cut -f1))"
fi

# ── INT8（无校准，PTQ 需额外数据，此处使用 FP16+INT8 混合模式）─────────────
ENGINE_INT8="${SCRIPT_DIR}/faster_rcnn_${HEIGHT}_int8.engine"
if [ -f "$ENGINE_INT8" ]; then
    echo "[SKIP] INT8 引擎已存在: $ENGINE_INT8"
else
    echo ""
    echo "[3/3] 构建 INT8 引擎（FP16 fallback，无校准表）..."
    echo "  注意：无校准数据的 INT8 可能导致精度下降，仅用于速度对比"
    $TRTEXEC \
        --onnx="$FIXED_ONNX" \
        --saveEngine="$ENGINE_INT8" \
        --fp16 --int8 \
        --minShapes="$SHAPES" \
        --optShapes="$SHAPES" \
        --maxShapes="$SHAPES" \
        --workspace=4096 || {
        echo "[WARN] INT8 构建失败（部分层不支持 INT8），跳过"
    }
fi

echo ""
echo "========================================"
echo " 构建完成！引擎文件列表："
ls -lh "${SCRIPT_DIR}"/faster_rcnn_${HEIGHT}*.engine 2>/dev/null
echo "========================================"
