#!/bin/bash
# =============================================================================
#  demo.sh ── Faster R-CNN ROS2 项目一键演示入口
# =============================================================================
#  用法：
#    bash demo.sh                    # 全流程演示（推荐）
#    bash demo.sh --no-infer         # 跳过实时推理，只展示已有结果
#    bash demo.sh --stage 2          # 只运行指定阶段
#    bash demo.sh --images 3         # 实时推理展示3张图
#    bash demo.sh --threshold 0.3    # 调整置信度阈值
#    bash demo.sh --help             # 查看帮助
# =============================================================================

set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS"

# ─────────────────────────── 颜色 ──────────────────────────────────────────
RED='\033[91m'; GREEN='\033[92m'; YELLOW='\033[93m'
CYAN='\033[96m'; BOLD='\033[1m'; RESET='\033[0m'
BG_BLUE='\033[44m'; WHITE='\033[97m'

info()  { echo -e "  ${CYAN}ℹ${RESET}  $*"; }
ok()    { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn()  { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
err()   { echo -e "  ${RED}✗${RESET}  $*" >&2; }
header(){ echo -e "\n${BG_BLUE}${WHITE}${BOLD}  $*  ${RESET}\n"; }

# ─────────────────────────── Banner ────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Faster R-CNN · ROS2 Foxy · TensorRT · Jetson AGX Orin     ║${RESET}"
echo -e "${BOLD}${CYAN}║                  一  键  演  示  脚  本                     ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ─────────────────────────── 帮助信息 ──────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "用法："
    echo "  bash demo.sh [选项]"
    echo ""
    echo "选项："
    echo "  --no-infer         跳过实时 TRT 推理，只展示已有结果"
    echo "  --stage N          只运行指定阶段 (1=架构, 2=推理, 3=精度, 4=可视化)"
    echo "  --images N         实时推理展示图片数量（默认 6）"
    echo "  --threshold F      检测置信度阈值（默认 0.5）"
    echo "  --output PATH      HTML 报告路径（默认 demo_report.html）"
    echo "  --no-clock         跳过 jetson_clocks 建议提示"
    echo "  --help             显示此帮助"
    echo ""
    echo "演示阶段说明："
    echo "  阶段 0  环境检查（始终运行）"
    echo "  阶段 1  项目架构图解"
    echo "  阶段 2  实时 TRT 推理演示（FP16/INT8）"
    echo "  阶段 3  精度对比表（四配置 × KITTI 100张）"
    echo "  阶段 4  检测结果可视化拼图"
    echo "  最终    生成自包含 HTML 培训报告"
    echo ""
    exit 0
fi

# ─────────────────────────── 参数解析 ──────────────────────────────────────
NO_CLOCK=0
PASS_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-clock) NO_CLOCK=1 ;;
        *) PASS_ARGS+=("$arg") ;;
    esac
done

# ─────────────────────────── 前置检查 ──────────────────────────────────────
header "前置环境检查"

# Python3
if ! command -v python3 &>/dev/null; then
    err "python3 未找到，请先安装 Python3"
    exit 1
fi
ok "python3: $(python3 --version)"

# 检查 Python 包
MISSING_PKGS=0
for pkg in tensorrt cv2 numpy; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        err "Python 包缺失: $pkg"
        MISSING_PKGS=$((MISSING_PKGS + 1))
    fi
done
if [[ $MISSING_PKGS -gt 0 ]]; then
    err "$MISSING_PKGS 个 Python 依赖缺失，请先安装后重试"
    exit 1
fi
ok "Python 依赖: tensorrt / cv2 / numpy ── 全部就绪"

# 引擎文件
MODELS_DIR="$WS/install/faster_rcnn_ros/share/faster_rcnn_ros/models"
ENGINE_FP16="$MODELS_DIR/faster_rcnn_500.engine"
ENGINE_INT8="$MODELS_DIR/faster_rcnn_500_int8.engine"

if [[ ! -f "$ENGINE_FP16" ]]; then
    err "FP16 引擎不存在: $ENGINE_FP16"
    err "请先在 Jetson 上运行以下命令构建引擎："
    echo -e "    ${YELLOW}python3 src/faster_rcnn_ros/models/build_engine.py --height 500 --width 1242${RESET}"
    exit 1
fi
ok "FP16 引擎: $ENGINE_FP16 ($(du -sh "$ENGINE_FP16" 2>/dev/null | cut -f1))"

if [[ -f "$ENGINE_INT8" ]]; then
    ok "INT8 引擎: $ENGINE_INT8 ($(du -sh "$ENGINE_INT8" 2>/dev/null | cut -f1))"
else
    warn "INT8 引擎不存在，精度对比表中 INT8 行将无推理数据"
fi

# 测试数据
IMG_COUNT=$(ls "$WS/test_images/kitti_100/images/"*.png 2>/dev/null | wc -l || echo 0)
if [[ "$IMG_COUNT" -lt 10 ]]; then
    warn "测试图片不足（当前 $IMG_COUNT 张），部分演示内容可能受影响"
else
    ok "测试图片: $IMG_COUNT 张"
fi

# GT 评估结果
GT_JSON="$WS/test_images/compare_results/gt_eval_summary.json"
if [[ -f "$GT_JSON" ]]; then
    ok "GT 评估结果: $GT_JSON"
else
    warn "未找到 gt_eval_summary.json，阶段3精度表将不可用"
    warn "可运行 'python3 gt_eval_all.py' 先评估（约 2 分钟）"
fi

echo ""

# ─────────────────────────── GPU 锁频建议 ──────────────────────────────────
if [[ $NO_CLOCK -eq 0 ]]; then
    CLOCKS_ON=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    if [[ "$CLOCKS_ON" != "performance" ]]; then
        echo -e "${YELLOW}  ┌─────────────────────────────────────────────────────────────┐${RESET}"
        echo -e "${YELLOW}  │  建议：运行 sudo jetson_clocks 锁定 GPU/CPU 频率              │${RESET}"
        echo -e "${YELLOW}  │  这将使推理延迟从 ~80ms 降至 ~63ms（约提升 20%）             │${RESET}"
        echo -e "${YELLOW}  │  执行：sudo jetson_clocks && bash demo.sh --no-clock         │${RESET}"
        echo -e "${YELLOW}  └─────────────────────────────────────────────────────────────┘${RESET}"
        echo ""
    else
        ok "GPU/CPU 已锁频（performance 模式）"
    fi
fi

# ─────────────────────────── Source ROS2 ──────────────────────────────────
header "加载 ROS2 环境"

if [[ -z "${ROS_DISTRO:-}" ]]; then
    for ros_dir in /opt/ros/foxy /opt/ros/humble /opt/ros/galactic; do
        if [[ -f "$ros_dir/setup.bash" ]]; then
            # shellcheck source=/dev/null
            set +u  # colcon setup.bash 使用了未绑定变量 COLCON_TRACE
            source "$ros_dir/setup.bash"
            set -u || true
            ok "已加载 ROS2: $ros_dir (ROS_DISTRO=$ROS_DISTRO)"
            break
        fi
    done
    if [[ -z "${ROS_DISTRO:-}" ]]; then
        warn "ROS2 未找到，某些功能将不可用"
    fi
else
    ok "ROS2 已加载 (ROS_DISTRO=$ROS_DISTRO)"
fi

if [[ -f "$WS/install/setup.bash" ]]; then
    set +u
    source "$WS/install/setup.bash"
    set -u || true
    ok "已加载工作空间: $WS/install/setup.bash"
fi

echo ""

# ─────────────────────────── 运行演示 ──────────────────────────────────────
header "启动演示"
echo -e "  命令: python3 demo_presentation.py ${PASS_ARGS[*]+${PASS_ARGS[*]}}"
echo ""

python3 "$WS/demo_presentation.py" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}

# ─────────────────────────── 完成提示 ──────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ══════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  演示完成！${RESET}"
echo -e "${GREEN}${BOLD}  ══════════════════════════════════════${RESET}"

if [[ -f "$WS/demo_report.html" ]]; then
    echo ""
    info "HTML 报告：file://$WS/demo_report.html"
    info "在浏览器中打开查看完整报告"
fi
if [[ -f "$WS/demo_mosaic.jpg" ]]; then
    info "可视化拼图：$WS/demo_mosaic.jpg"
fi
echo ""
