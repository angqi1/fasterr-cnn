#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_kitti_eval.sh — KITTI Recall 验证一键脚本
#
# 用法:
#   bash run_kitti_eval.sh [选项]
#
# 选项:
#   --count N         下载图片张数 (默认 100)
#   --input-h H       推理输入高度 (默认 500, 375 或 500)
#   --threshold T     检测置信度阈值 (默认 0.25)
#   --iou-thresh I    命中判定 IoU 阈值 (默认 0.3)
#   --out-root DIR    输出根目录 (默认 test_images/kitti_eval)
#   --onnx PATH       ONNX 模型路径 (默认自动)
#   --skip-download   跳过下载(已有数据时使用)
#   -h, --help        显示帮助
#
# 快速示例:
#   bash run_kitti_eval.sh                          # 下载100张,500高度评估
#   bash run_kitti_eval.sh --count 200              # 下载200张
#   bash run_kitti_eval.sh --skip-download --input-h 375   # 仅重跑评估
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── 默认参数 ────────────────────────────────────────────────────────────────
COUNT=100
INPUT_H=500
THRESHOLD=0.25
IOU_THRESH=0.3
OUT_ROOT=""
ONNX=""
SKIP_DOWNLOAD=0

PYTHON=/usr/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 参数解析 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --count)       COUNT="$2";       shift 2 ;;
        --input-h)     INPUT_H="$2";     shift 2 ;;
        --threshold)   THRESHOLD="$2";   shift 2 ;;
        --iou-thresh)  IOU_THRESH="$2";  shift 2 ;;
        --out-root)    OUT_ROOT="$2";     shift 2 ;;
        --onnx)        ONNX="$2";         shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 自动设默认值 ────────────────────────────────────────────────────────────
[[ -z "$OUT_ROOT" ]] && OUT_ROOT="${SCRIPT_DIR}/test_images/kitti_${COUNT}"
[[ -z "$ONNX" ]]     && ONNX="${SCRIPT_DIR}/src/faster_rcnn_ros/models/fasterrcnn_nuscenes.onnx"

IMG_DIR="${OUT_ROOT}/images"
LBL_DIR="${OUT_ROOT}/labels"
VIS_DIR="${OUT_ROOT}/results_vis_h${INPUT_H}"
REPORT="${OUT_ROOT}/metrics_h${INPUT_H}.txt"

# ── 打印配置 ────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════╗"
echo "║    KITTI Recall 验证流程            ║"
echo "╚══════════════════════════════════════╝"
echo "  图片数量  : $COUNT"
echo "  输入高度  : $INPUT_H"
echo "  置信阈值  : $THRESHOLD"
echo "  IoU 阈值  : $IOU_THRESH"
echo "  数据目录  : $OUT_ROOT"
echo "  ONNX 模型 : $ONNX"
echo "  跳过下载  : $([ $SKIP_DOWNLOAD -eq 1 ] && echo '是' || echo '否')"
echo ""

# ── Step 1: 下载 ─────────────────────────────────────────────────────────────
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    echo "▶ [1/2] 从 KITTI 官方下载 ${COUNT} 张样本..."
    $PYTHON "${SCRIPT_DIR}/download_kitti_subset.py" \
        --count "$COUNT" \
        --out-root "$OUT_ROOT"

    N_IMG=$(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l)
    N_LBL=$(ls "$LBL_DIR"/*.txt 2>/dev/null | wc -l)
    echo "  ✓ 图片: $N_IMG, 标签: $N_LBL"
else
    N_IMG=$(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l)
    echo "▶ [1/2] 跳过下载，使用已有数据 ($N_IMG 张)"
fi

echo ""

# ── Step 2: 推理评估 ─────────────────────────────────────────────────────────
echo "▶ [2/2] 推理评估 (input_h=$INPUT_H)..."
$PYTHON "${SCRIPT_DIR}/compare_gt.py" \
    --img-dir "$IMG_DIR" \
    --gt-dir  "$LBL_DIR" \
    --out-dir "$VIS_DIR" \
    --onnx    "$ONNX" \
    --input-h "$INPUT_H" \
    --threshold  "$THRESHOLD" \
    --iou-thresh "$IOU_THRESH" \
    | tee "$REPORT"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║    完成                             ║"
echo "╚══════════════════════════════════════╝"
echo "  评估报告 : $REPORT"
echo "  可视化图 : $VIS_DIR/cmp_*.png"
