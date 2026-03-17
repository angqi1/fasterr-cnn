#!/usr/bin/env python3
"""
analyze_detection.py
对指定图片用 ONNX Runtime 推理，多阈值扫描，与 GT 对比可视化。

用法:
  python3 analyze_detection.py
"""
import cv2
import numpy as np
import onnxruntime as ort

# ─── 配置 ──────────────────────────────────────────────────────────────────
IMG_PATH   = "test_images/5A259630E4A436EFC07FDCBD839630FB.png"
ONNX_PATH  = "src/faster_rcnn_ros/models/fasterrcnn_nuscenes.onnx"
OUT_DIR    = "test_images/results_cpp"
INPUT_H, INPUT_W = 375, 1242

LABEL_MAP = {
    0: "background", 1: "car", 2: "truck", 3: "bus", 4: "trailer",
    5: "construction_vehicle", 6: "pedestrian",
    7: "motorcycle", 8: "bicycle", 9: "traffic_cone",
}

# KITTI 格式 GT (left, top, right, bottom) / 类别
# truncated occluded 状态标注在注释中
GT_BOXES = [
    # (class, left, top, right, bottom, trunc, occl)
    ("car",  572.31, 171.60, 624.11, 218.82, 0.00, 0),  # GT1
    ("car",    0.00, 190.67, 171.41, 313.27, 0.43, 0),  # GT2  左截断
    ("car",  214.99, 165.20, 423.55, 292.98, 0.00, 0),  # GT3  大框
    ("car",  296.94, 182.19, 393.12, 231.61, 0.00, 1),  # GT4  遮挡
    ("car",  428.19, 180.81, 498.01, 226.24, 0.00, 0),  # GT5
    ("van",  481.87, 153.05, 537.09, 207.74, 0.00, 1),  # GT6  Van+遮挡
    ("car",  522.27, 176.55, 552.71, 202.38, 0.00, 1),  # GT7  小框+遮挡
    ("car",    0.00, 185.88, 180.39, 253.13, 0.04, 1),  # GT8  左截断+遮挡
    ("car",  193.59, 181.29, 294.71, 229.39, 0.00, 2),  # GT9  严重遮挡
]

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

# 颜色: GT=绿色, pred=渐变(蓝→红)
GT_COLOR   = (0, 200, 0)
PRED_COLORS = {
    0.1: (255, 50,  50),   # 蓝偏红
    0.2: (50,  50, 255),   # 红
    0.3: (0,  200, 200),   # 黄
    0.4: (200,  0, 200),   # 紫
    0.5: (0,  140, 255),   # 橙
}

# ─── 推理 ──────────────────────────────────────────────────────────────────
def preprocess(img_bgr):
    """resize → RGB → /255 → CHW float32"""
    resized = cv2.resize(img_bgr, (INPUT_W, INPUT_H))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw     = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return chw[np.newaxis, ...]   # (1, 3, H, W)

def run_ort(session, img_bgr):
    inp = preprocess(img_bgr)
    boxes, labels, scores = session.run(
        ["boxes", "labels", "scores"],
        {"image": inp}
    )
    return boxes, labels.astype(int), scores

def iou(a, b):
    """IoU between box a=[x1,y1,x2,y2] and b=[x1,y1,x2,y2]"""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# ─── 可视化单阈值 ───────────────────────────────────────────────────────────
def draw_analysis(img_bgr, boxes, labels, scores, threshold, scale_x, scale_y):
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # 画 GT（绿色虚线框）
    for i, (cls, x1, y1, x2, y2, trunc, occl) in enumerate(GT_BOXES):
        # GT 坐标基于原始图像尺寸，需要缩放
        px1 = int(x1 * scale_x); py1 = int(y1 * scale_y)
        px2 = int(x2 * scale_x); py2 = int(y2 * scale_y)
        cv2.rectangle(vis, (px1, py1), (px2, py2), GT_COLOR, 2)
        occ_str = f"trunc={trunc:.2f} occ={occl}"
        cv2.putText(vis, f"GT{i+1}:{cls} {occ_str}",
                    (max(0,px1), max(12, py1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GT_COLOR, 1)

    # 画预测框（红色实线框）
    pred_color = PRED_COLORS.get(threshold, (0, 0, 255))
    detected = [(b, l, s) for b, l, s in zip(boxes, labels, scores) if s >= threshold]
    for b, l, s in detected:
        # boxes 坐标是基于 INPUT_H×INPUT_W 的推理尺寸，需要缩放回显示尺寸
        px1 = int(b[0] * scale_x); py1 = int(b[1] * scale_y)
        px2 = int(b[2] * scale_x); py2 = int(b[3] * scale_y)
        cv2.rectangle(vis, (px1, py1), (px2, py2), pred_color, 2)
        lname = LABEL_MAP.get(l, str(l))
        cv2.putText(vis, f"{lname} {s:.2f}",
                    (px1, max(12, py1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1)

    # 标题
    cv2.putText(vis, f"thresh={threshold}  pred={len(detected)}/9 GT",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"thresh={threshold}  pred={len(detected)}/9 GT",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return vis

# ─── IoU 匹配分析 ──────────────────────────────────────────────────────────
def match_analysis(boxes_onnx, labels, scores, threshold, orig_w, orig_h):
    """将推理框（INPUT 坐标系）映射回原图坐标，然后与 GT 计算 IoU"""
    sx = orig_w / INPUT_W
    sy = orig_h / INPUT_H

    print(f"\n──── threshold = {threshold} ────")
    dets = [(b, l, s) for b, l, s in zip(boxes_onnx, labels, scores) if s >= threshold]
    print(f"检测到 {len(dets)} 个目标:")
    for b, l, s in dets:
        # 转回原图坐标
        rb = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
        lname = LABEL_MAP.get(l, str(l))
        print(f"  [{lname:20s}] score={s:.3f}  box=({rb[0]:.0f},{rb[1]:.0f},{rb[2]:.0f},{rb[3]:.0f})")

    # 匹配每个 GT
    print(f"\nGT 匹配情况 (IoU≥0.3):")
    for i, (cls, x1, y1, x2, y2, trunc, occl) in enumerate(GT_BOXES):
        gt_box = [x1, y1, x2, y2]
        best_iou  = 0.0
        best_det  = None
        for b, l, s in dets:
            rb = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
            iou_val = iou(gt_box, rb)
            if iou_val > best_iou:
                best_iou = iou_val
                best_det = (l, s, rb)
        status = "✅ 命中" if best_iou >= 0.3 else "❌ 漏检"
        occ_info = f"trunc={trunc:.2f} occ={occl}"
        if best_det:
            lname = LABEL_MAP.get(best_det[0], str(best_det[0]))
            print(f"  GT{i+1} [{cls:4s}] {occ_info:20s} → {status} "
                  f"IoU={best_iou:.2f} pred=[{lname} {best_det[1]:.2f}]")
        else:
            print(f"  GT{i+1} [{cls:4s}] {occ_info:20s} → {status}")

# ─── All-scores 直方图分析 ─────────────────────────────────────────────────
def score_histogram(scores):
    print("\n──── 全部检测分数分布 (无阈值) ────")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(scores, bins=bins)
    total = len(scores)
    print(f"总候选框: {total}")
    for lo, hi, cnt in zip(bins[:-1], bins[1:], counts):
        bar = "█" * (cnt * 40 // max(1, total))
        print(f"  [{lo:.1f}-{hi:.1f}): {cnt:4d}  {bar}")

# ─── main ──────────────────────────────────────────────────────────────────
def main():
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"图片未找到: {IMG_PATH}")
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"原图尺寸: {orig_w}×{orig_h}")
    print(f"GT 共 {len(GT_BOXES)} 个目标")

    # 加载 ONNX Session
    print(f"\n加载 ONNX: {ONNX_PATH}")
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # 只显示错误
    session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options,
                                   providers=["CPUExecutionProvider"])
    print("推理 (CPU ORT, 可能稍慢)...")
    boxes, labels, scores = run_ort(session, img_bgr)
    print(f"原始输出框数: {len(scores)}")

    # 分数直方图
    score_histogram(scores)

    # 各阈值匹配分析
    for thr in THRESHOLDS:
        match_analysis(boxes, labels, scores, thr, orig_w, orig_h)

    # 显示尺寸 = 原图尺寸（boxes 坐标基于 INPUT_H×INPUT_W，需要缩放显示）
    scale_x = orig_w / INPUT_W
    scale_y = orig_h / INPUT_H

    # ── 生成可视化图：每个阈值一张 ──────────────────────────────────────────
    vis_list = []
    for thr in THRESHOLDS:
        vis = draw_analysis(img_bgr, boxes, labels, scores, thr, scale_x, scale_y)
        out_path = f"{OUT_DIR}/analysis_thr{int(thr*100):02d}.jpg"
        cv2.imwrite(out_path, vis)
        vis_list.append(vis)

    # ── 合并成竖向拼图 ───────────────────────────────────────────────────────
    combined = np.vstack(vis_list)
    combined_path = f"{OUT_DIR}/analysis_combined.jpg"
    cv2.imwrite(combined_path, combined)
    print(f"\n✅ 合并可视化图: {combined_path}")

    # ── GT only 参考图 ───────────────────────────────────────────────────────
    gt_vis = img_bgr.copy()
    for i, (cls, x1, y1, x2, y2, trunc, occl) in enumerate(GT_BOXES):
        cv2.rectangle(gt_vis, (int(x1*scale_x), int(y1*scale_y)),
                               (int(x2*scale_x), int(y2*scale_y)), GT_COLOR, 2)
        cv2.putText(gt_vis, f"GT{i+1}:{cls}",
                    (max(0, int(x1*scale_x)), max(15, int(y1*scale_y)-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GT_COLOR, 1)
    cv2.imwrite(f"{OUT_DIR}/gt_reference.jpg", gt_vis)
    print(f"GT 参考图: {OUT_DIR}/gt_reference.jpg")

    # ── 打印漏检汇总（阈值 0.4）────────────────────────────────────────────
    print("\n══════ 漏检分析汇总 (threshold=0.4) ══════")
    dets04 = [(b, l, s) for b, l, s in zip(boxes, labels, scores) if s >= 0.4]
    sx = orig_w / INPUT_W; sy = orig_h / INPUT_H
    missed = []
    for i, (cls, x1, y1, x2, y2, trunc, occl) in enumerate(GT_BOXES):
        gt_box = [x1, y1, x2, y2]
        best_iou = max((iou(gt_box, [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy])
                       for b, l, s in dets04), default=0.0)
        # 找最高 score（无论阈值）
        best_score = max((s for b, l, s in zip(boxes, labels, scores)
                          if iou(gt_box, [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]) >= 0.3),
                         default=0.0)
        area = (x2-x1)*(y2-y1)
        hit  = best_iou >= 0.3
        if not hit:
            missed.append(i+1)
        status = "✅" if hit else "❌"
        reason = ""
        if not hit:
            if trunc >= 0.3: reason += " [左边缘截断]"
            if occl >= 2:    reason += " [严重遮挡]"
            if occl >= 1:    reason += " [部分遮挡]"
            if area < 3000:  reason += " [框面积小]"
            if best_score > 0: reason += f" [最高分={best_score:.3f}]"
            else:              reason += " [完全未响应]"
        print(f"  GT{i+1} [{cls:3s}] trunc={trunc:.2f} occ={occl} "
              f"area={area:.0f}px²  → {status}{reason}")

    if missed:
        print(f"\n漏检 GT 编号: {missed}")
    else:
        print("\n全部命中！")

if __name__ == "__main__":
    main()
