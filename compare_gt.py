#!/usr/bin/env python3
"""
compare_gt.py
用 ONNX Runtime 推理 test_images/ 下所有图片，与 results/ 中的 KITTI 标注对比，
生成每张图的可视化对比图和汇总统计。

用法:
  cd /home/nvidia/ros2_ws
  python3 compare_gt.py
"""
import os
import argparse
import cv2
import numpy as np
import onnxruntime as ort

# ─── 默认配置（可通过命令行覆盖）──────────────────────────────────────────────
IMG_DIR    = "test_images"
GT_DIR     = "test_images/results"
OUT_DIR    = "test_images/results_cpp"
ONNX_PATH  = "src/faster_rcnn_ros/models/fasterrcnn_nuscenes.onnx"
INPUT_H, INPUT_W = 375, 1242
THRESHOLD  = 0.3
IOU_THRESH = 0.3   # 命中判定 IoU 阈值

# NuScenes 类别（与 labels.txt 一致）
LABEL_MAP = {
    0: "background", 1: "car", 2: "truck", 3: "bus", 4: "trailer",
    5: "construction_vehicle", 6: "pedestrian",
    7: "motorcycle", 8: "bicycle", 9: "traffic_cone",
}

# KITTI 类别 → NuScenes 类别 映射（用于 IoU 匹配时的类别宽容度）
KITTI_TO_NUSCENES = {
    "car": [1], "van": [1, 2], "truck": [2], "bus": [3],
    "pedestrian": [6], "person_sitting": [6],
    "cyclist": [7, 8], "tram": [3],
    "misc": list(range(1, 10)),
    "dontcare": [],
}

GT_COLOR   = (0, 210, 0)      # 绿色：GT
PRED_COLOR = (0, 100, 255)    # 橙色：预测


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_kitti_gt(txt_path):
    """解析 KITTI 标注，返回 [(class_str, x1, y1, x2, y2, truncated, occluded)]"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls   = parts[0].lower()
            trunc = float(parts[1])
            occl  = int(parts[2])
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            boxes.append((cls, x1, y1, x2, y2, trunc, occl))
    return boxes


def preprocess(img_bgr):
    resized = cv2.resize(img_bgr, (INPUT_W, INPUT_H))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[np.newaxis]


def run_ort(session, img_bgr):
    boxes, labels, scores = session.run(
        ["boxes", "labels", "scores"], {"image": preprocess(img_bgr)}
    )
    return boxes, labels.astype(int), scores


def match_preds_to_gt(gt_boxes, pred_boxes, pred_labels, pred_scores,
                      orig_w, orig_h, threshold):
    """
    将预测框（INPUT 坐标）转回原图坐标，然后与 GT 贪心匹配。
    返回:
      hits   : GT 中被命中的索引集合
      fp_list: 假阳性预测框列表 [(box, label, score)]
    """
    sx = orig_w / INPUT_W
    sy = orig_h / INPUT_H

    dets = [(b, l, s) for b, l, s in zip(pred_boxes, pred_labels, pred_scores)
            if s >= threshold]

    # 对 GT 排优先级（按面积降序），避免大框抢走小框的匹配
    gt_with_idx = [(i, g) for i, g in enumerate(gt_boxes) if g[0] != "dontcare"]
    gt_with_idx.sort(key=lambda x: -(x[1][3]-x[1][1])*(x[1][4]-x[1][2]))

    matched_gt   = set()
    matched_pred = set()

    for gi, (cls, x1, y1, x2, y2, trunc, occl) in gt_with_idx:
        gt_box = [x1, y1, x2, y2]
        best_iou = 0.0
        best_di  = -1
        for di, (b, l, s) in enumerate(dets):
            if di in matched_pred:
                continue
            rb = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
            iv = iou(gt_box, rb)
            if iv > best_iou:
                best_iou = iv
                best_di  = di
        if best_iou >= IOU_THRESH:
            matched_gt.add(gi)
            matched_pred.add(best_di)

    fp_list = [(dets[di][0], dets[di][1], dets[di][2])
               for di in range(len(dets)) if di not in matched_pred]
    return matched_gt, fp_list


def draw_comparison(img_bgr, gt_boxes, pred_boxes, pred_labels, pred_scores,
                    threshold, orig_w, orig_h, matched_gt):
    """绘制 GT（绿框）和预测（橙框）的对比图。"""
    vis = img_bgr.copy()
    sx  = orig_w / INPUT_W
    sy  = orig_h / INPUT_H

    # GT 框
    for i, (cls, x1, y1, x2, y2, trunc, occl) in enumerate(gt_boxes):
        if cls == "dontcare":
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (128, 128, 128), 1)
            continue
        color = GT_COLOR if i in matched_gt else (0, 0, 200)  # 红=漏检
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label_str = f"GT:{cls[:4]}"
        if occl > 0: label_str += f" o{occl}"
        cv2.putText(vis, label_str,
                    (max(0, int(x1)), max(12, int(y1)-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # 预测框
    for b, l, s in zip(pred_boxes, pred_labels, pred_scores):
        if s < threshold:
            continue
        px1 = int(b[0]*sx); py1 = int(b[1]*sy)
        px2 = int(b[2]*sx); py2 = int(b[3]*sy)
        cv2.rectangle(vis, (px1, py1), (px2, py2), PRED_COLOR, 2)
        lname = LABEL_MAP.get(l, str(l))
        cv2.putText(vis, f"{lname[:4]} {s:.2f}",
                    (px1, max(12, py1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, PRED_COLOR, 1)

    n_gt   = sum(1 for g in gt_boxes if g[0] != "dontcare")
    n_pred = sum(1 for s in pred_scores if s >= threshold)
    n_hit  = len(matched_gt)
    cv2.putText(vis, f"GT={n_gt}  Pred={n_pred}  Hit={n_hit}  Miss={n_gt-n_hit}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(vis, f"GT={n_gt}  Pred={n_pred}  Hit={n_hit}  Miss={n_gt-n_hit}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    return vis


# ─── 主逻辑 ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default=IMG_DIR)
    parser.add_argument("--gt-dir", default=GT_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--onnx", default=ONNX_PATH)
    parser.add_argument("--input-h", type=int, default=INPUT_H)
    parser.add_argument("--input-w", type=int, default=INPUT_W)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--iou-thresh", type=float, default=IOU_THRESH)
    return parser.parse_args()


def main():
    global IMG_DIR, GT_DIR, OUT_DIR, ONNX_PATH
    global INPUT_H, INPUT_W, THRESHOLD, IOU_THRESH

    args = parse_args()
    IMG_DIR = args.img_dir
    GT_DIR = args.gt_dir
    OUT_DIR = args.out_dir
    ONNX_PATH = args.onnx
    INPUT_H = args.input_h
    INPUT_W = args.input_w
    THRESHOLD = args.threshold
    IOU_THRESH = args.iou_thresh

    os.makedirs(OUT_DIR, exist_ok=True)

    # 加载 ONNX session
    print(f"加载 ONNX: {ONNX_PATH}")
    opts = ort.SessionOptions(); opts.log_severity_level = 3
    session = ort.InferenceSession(ONNX_PATH, sess_options=opts,
                                   providers=["CPUExecutionProvider"])

    # 收集图片（仅 .png/.jpg，排除子目录）
    img_files = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg")) and
           os.path.isfile(os.path.join(IMG_DIR, f))
    ])
    print(f"共 {len(img_files)} 张图片，threshold={THRESHOLD}\n")

    # 汇总统计
    total_gt = total_hit = total_pred = total_fp = 0
    per_class_gt  = {}
    per_class_hit = {}

    print(f"{'图片':<14}  {'GT':>4}  {'Pred':>5}  {'Hit':>4}  {'Miss':>5}  {'FP':>4}  {'Rec%':>6}")
    print("─" * 60)

    for fname in img_files:
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(IMG_DIR, fname)
        gt_path  = os.path.join(GT_DIR, stem + ".txt")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        gt_boxes = parse_kitti_gt(gt_path)
        boxes, labels, scores = run_ort(session, img_bgr)

        gt_valid = [g for g in gt_boxes if g[0] != "dontcare"]
        matched_gt, fp_list = match_preds_to_gt(
            gt_boxes, boxes, labels, scores, orig_w, orig_h, THRESHOLD)

        n_gt   = len(gt_valid)
        n_pred = sum(1 for s in scores if s >= THRESHOLD)
        n_hit  = len(matched_gt)
        n_miss = n_gt - n_hit
        n_fp   = len(fp_list)
        recall = n_hit / n_gt * 100 if n_gt > 0 else float("nan")

        total_gt   += n_gt
        total_hit  += n_hit
        total_pred += n_pred
        total_fp   += n_fp

        # 按类别统计
        for i, (cls, *_) in enumerate(gt_valid):
            per_class_gt[cls]  = per_class_gt.get(cls, 0) + 1
            if i in matched_gt:
                per_class_hit[cls] = per_class_hit.get(cls, 0) + 1

        rec_str = f"{recall:.0f}%" if not (recall != recall) else " N/A"
        print(f"{fname:<14}  {n_gt:>4}  {n_pred:>5}  {n_hit:>4}  {n_miss:>5}  {n_fp:>4}  {rec_str:>6}")

        # 生成可视化对比图
        vis = draw_comparison(img_bgr, gt_boxes, boxes, labels, scores,
                              THRESHOLD, orig_w, orig_h, matched_gt)
        out_path = os.path.join(OUT_DIR, f"cmp_{fname}")
        cv2.imwrite(out_path, vis)

    # ─── 汇总 ───────────────────────────────────────────────────────────────
    print("─" * 60)
    overall_rec = total_hit / total_gt * 100 if total_gt > 0 else 0
    precision   = total_hit / total_pred * 100 if total_pred > 0 else 0
    f1          = (2 * total_hit / (total_gt + total_pred) * 100
                   if (total_gt + total_pred) > 0 else 0)
    print(f"{'合计':<14}  {total_gt:>4}  {total_pred:>5}  {total_hit:>4}  "
          f"{total_gt-total_hit:>5}  {total_fp:>4}  {overall_rec:>5.1f}%")
    print(f"\n整体指标 (IoU≥{IOU_THRESH}, threshold={THRESHOLD}):")
    print(f"  Recall    : {total_hit}/{total_gt} = {overall_rec:.1f}%")
    print(f"  Precision : {total_hit}/{total_pred} = {precision:.1f}%")
    print(f"  F1-score  : {f1:.1f}%")

    print(f"\n按类别召回率:")
    for cls in sorted(per_class_gt.keys()):
        gt_c  = per_class_gt[cls]
        hit_c = per_class_hit.get(cls, 0)
        rec_c = hit_c / gt_c * 100
        bar   = "█" * int(rec_c / 5)
        print(f"  {cls:<24} {hit_c:>3}/{gt_c:<3}  {rec_c:>5.1f}%  {bar}")

    print(f"\n✅ 结果图已保存至 {OUT_DIR}/cmp_*.png")


if __name__ == "__main__":
    main()
