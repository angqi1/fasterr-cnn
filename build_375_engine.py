#!/usr/bin/env python3
"""
从 fasterrcnn_nuscenes_fixed_500x1242.onnx 构建 375×1242 引擎。

500x1242 TopK K 值为 [1000,1000,1000,K3=540,K4=150]。
在 375 高度下，TRT 报告 TopK_3 axis=420 < K=540，需要修正。
策略：
  - 运行 trtexec 收集所有 TopK 的实际 axis 大小（从错误信息中提取）
  - 针对实际超限的节点修正 K = axis_size
"""

import os, sys, subprocess, re
import numpy as np
import onnx
import onnx.numpy_helper as nph

FIXED_500_ONNX = "src/faster_rcnn_ros/models/fasterrcnn_nuscenes_fixed_500x1242.onnx"
FIXED_375_ONNX = "src/faster_rcnn_ros/models/fasterrcnn_nuscenes_fixed_375x1242.onnx"
OUTPUT_ENGINE  = "install/faster_rcnn_ros/share/faster_rcnn_ros/models/faster_rcnn_375.engine"
TRTEXEC        = "/usr/src/tensorrt/bin/trtexec"

H, W = 375, 1242


def get_topk_k_map(onnx_path):
    """返回 {k_init_name: current_k} 和 {topk_node_name: k_init_name}"""
    model = onnx.load(onnx_path)
    init_map = {i.name: nph.to_array(i) for i in model.graph.initializer}
    k_map = {}
    node_k_map = {}
    for node in model.graph.node:
        if node.op_type == "TopK":
            k_input = node.input[1]
            if k_input in init_map:
                k_map[k_input] = int(init_map[k_input].flatten()[0])
                node_k_map[node.name] = k_input
    return k_map, node_k_map


def run_trtexec_dry(onnx_path, h, w):
    """运行 trtexec 收集 axis < K 的错误，返回 {node_name: axis_size}"""
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        "--saveEngine=/dev/null",
        "--fp16",
        f"--minShapes=image:1x3x{h}x{w}",
        f"--optShapes=image:1x3x{h}x{w}",
        f"--maxShapes=image:1x3x{h}x{w}",
        "--workspace=4096",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    # 解析错误: "(/rpn/TopK_n: length of reduction axis (NNN) is smaller than K (MMM))"
    axis_map = {}
    for m in re.finditer(
        r"\((/rpn/TopK[^:]*): length of reduction axis \((\d+)\) is smaller than K \((\d+)\)",
        output
    ):
        node_name, axis_size, k_val = m.group(1), int(m.group(2)), int(m.group(3))
        axis_map[node_name] = axis_size
        print(f"  TRT 报告: {node_name}  axis={axis_size}  K={k_val}  → 需修正为 {axis_size}")

    return axis_map


def patch_topk(src_onnx, dst_onnx, node_k_map, k_map, axis_map):
    """将超限的 TopK K → axis_size，写入新 ONNX"""
    model = onnx.load(src_onnx)
    patched = 0
    for node_name, axis_size in axis_map.items():
        k_init_name = node_k_map.get(node_name)
        if k_init_name is None:
            print(f"  警告: 找不到 {node_name} 对应的 K initializer")
            continue
        for init in model.graph.initializer:
            if init.name == k_init_name:
                old_k = k_map[k_init_name]
                new_k = axis_size
                arr = np.array([new_k], dtype=np.int64)
                init.CopyFrom(nph.from_array(arr, name=init.name))
                print(f"  修正 {node_name}: K {old_k} → {new_k}")
                patched += 1
                break
    print(f"  共修正 {patched} 个 TopK K 值")
    onnx.save(model, dst_onnx)
    print(f"  已保存: {dst_onnx}")


def build_engine(onnx_path, engine_path, h, w):
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes=image:1x3x{h}x{w}",
        f"--optShapes=image:1x3x{h}x{w}",
        f"--maxShapes=image:1x3x{h}x{w}",
        "--workspace=4096",
    ]
    print(f"\n  CMD: {' '.join(cmd[-4:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    tail = lambda s: "\n".join(s.splitlines()[-20:]) if s else ""
    if result.returncode != 0:
        print("=== 构建失败 STDERR ===\n" + tail(result.stderr))
        print("=== STDOUT ===\n" + tail(result.stdout))
        return False

    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"✅ 引擎构建成功: {engine_path} ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    print(f"=== 构建 {H}×{W} FP16 引擎 ===\n")

    k_map, node_k_map = get_topk_k_map(FIXED_500_ONNX)
    print(f"原始 500h TopK K 值: {k_map}")

    current_onnx = FIXED_500_ONNX
    max_iter = 5  # 最多迭代 5 次修正

    for iteration in range(max_iter):
        print(f"\n[试运行 {iteration+1}] 检测 {H}×{W} 下的 TopK 超限 ...")
        axis_map = run_trtexec_dry(current_onnx, H, W)

        if not axis_map:
            print("  没有 TopK 超限，准备正式构建")
            break

        # 修正并使用新 ONNX 继续
        out_onnx = FIXED_375_ONNX
        k_map_cur, node_k_map_cur = get_topk_k_map(current_onnx)
        patch_topk(current_onnx, out_onnx, node_k_map_cur, k_map_cur, axis_map)
        current_onnx = out_onnx
    else:
        print("超过最大迭代次数，终止")
        sys.exit(1)

    print(f"\n[正式构建] {current_onnx} → {OUTPUT_ENGINE} ...")
    if not build_engine(current_onnx, OUTPUT_ENGINE, H, W):
        sys.exit(1)

"""
从 fasterrcnn_nuscenes_fixed_500x1242.onnx 构建 375×1242 引擎。

500x1242 引擎的 TopK K 值是 [1000,1000,1000,540,150]，
在 375 高度下 P5/P6 级别的锚框数量可能小于这些 K 值，需先修正。
"""

import os, sys, subprocess
import numpy as np
import onnx
import onnx.numpy_helper as nph
import onnxruntime as ort
import io

FIXED_500_ONNX = "src/faster_rcnn_ros/models/fasterrcnn_nuscenes_fixed_500x1242.onnx"
FIXED_375_ONNX = "src/faster_rcnn_ros/models/fasterrcnn_nuscenes_fixed_375x1242.onnx"
OUTPUT_ENGINE  = "install/faster_rcnn_ros/share/faster_rcnn_ros/models/faster_rcnn_375.engine"
TRTEXEC        = "/usr/src/tensorrt/bin/trtexec"

H, W = 375, 1242

def compute_topk_k_for_375(fixed_onnx_path, h, w):
    """通过 ORT 推理获取每个 TopK 节点数据输入的实际维度大小。"""
    model = onnx.load(fixed_onnx_path)
    
    # 找到所有 TopK 节点及其数据输入名称
    topk_data_inputs = {}
    for node in model.graph.node:
        if node.op_type == "TopK":
            topk_data_inputs[node.name] = node.input[0]
    
    print(f"TopK 节点: {list(topk_data_inputs.keys())}")
    
    # 添加额外的图输出来观察 TopK 数据输入的 shape
    existing_outputs = {o.name for o in model.graph.output}
    for tname, data_in in topk_data_inputs.items():
        if data_in not in existing_outputs:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(data_in, onnx.TensorProto.FLOAT, None)
            )
    
    buf = io.BytesIO()
    onnx.save(model, buf)
    buf.seek(0)
    
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(buf.read(), sess_options=so,
                                providers=["CPUExecutionProvider"])
    
    fake = np.zeros((1, 3, h, w), dtype=np.float32)
    out_names = [o.name for o in sess.get_outputs()]
    
    # 只运行我们添加的额外输出
    extra_names = [data_in for data_in in topk_data_inputs.values()
                   if data_in in out_names]
    print(f"  运行 ORT at {h}×{w}，提取 {len(extra_names)} 个 TopK 输入 shape ...")
    results = sess.run(extra_names, {"image": fake})
    
    # results[i] 对应 extra_names[i]
    size_map = {}
    for name, arr in zip(extra_names, results):
        # TopK 通常在最后一维操作，记录 flat 长度
        size_map[name] = arr.flatten().shape[0]
        print(f"    {name}: shape={arr.shape}, flat_size={arr.flatten().shape[0]}")
    
    # 获取各 TopK 节点对应的正确 K 值
    # 原始模型中 TopK K 由 initializer 直接给出
    init_map = {i.name: nph.to_array(i) for i in model.graph.initializer}
    
    # 重新加载原始 fixed_500 来获取 topk_k_const 初始值
    model_orig = onnx.load(fixed_onnx_path)
    init_map_orig = {i.name: nph.to_array(i) for i in model_orig.graph.initializer}
    
    corrected_k = {}
    for node in model_orig.graph.node:
        if node.op_type == "TopK":
            k_input = node.input[1]
            data_input = node.input[0]
            orig_k = int(init_map_orig[k_input].flatten()[0])
            actual_size = size_map.get(data_input, orig_k)
            new_k = min(orig_k, actual_size)
            corrected_k[k_input] = (orig_k, new_k)
            marker = "✓" if new_k == orig_k else f"⚠ 修正 {orig_k}→{new_k}"
            print(f"    TopK={node.name}  K_init={k_input}  orig_K={orig_k}  axis_size={actual_size}  new_K={new_k}  {marker}")
    
    return corrected_k


def patch_onnx_topk(fixed_onnx_path, out_onnx_path, corrected_k):
    """将修正后的 K 值写入 ONNX 的 initializer。"""
    model = onnx.load(fixed_onnx_path)
    patched = 0
    for init in model.graph.initializer:
        if init.name in corrected_k:
            orig_k, new_k = corrected_k[init.name]
            if new_k != orig_k:
                arr = np.array([new_k], dtype=np.int64)
                init.CopyFrom(nph.from_array(arr, name=init.name))
                print(f"  Patched {init.name}: {orig_k} → {new_k}")
                patched += 1
    print(f"  共修正 {patched} 个 TopK K 值")
    
    # 更新输入形状为当前 H×W（可选，不影响 trtexec）
    onnx.save(model, out_onnx_path)
    print(f"  保存至 {out_onnx_path}")


def build_engine(onnx_path, engine_path, h, w):
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes=image:1x3x{h}x{w}",
        f"--optShapes=image:1x3x{h}x{w}",
        f"--maxShapes=image:1x3x{h}x{w}",
        "--workspace=4096",
    ]
    print(f"\n[trtexec] {' '.join(cmd[-5:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    
    tail = lambda s: s[-3000:] if len(s) > 3000 else s
    if result.stdout:
        print("=== STDOUT ===\n" + tail(result.stdout))
    if result.stderr:
        print("=== STDERR ===\n" + tail(result.stderr))
    
    if result.returncode == 0:
        size_mb = os.path.getsize(engine_path) / 1024 / 1024
        print(f"\n✅ 引擎构建成功: {engine_path} ({size_mb:.1f} MB)")
    else:
        print(f"\n❌ 引擎构建失败 (returncode={result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    print(f"=== 构建 {H}×{W} FP16 引擎 ===\n")
    
    print(f"[1] 分析 {H}×{W} 下各 TopK 的实际轴大小 ...")
    corrected_k = compute_topk_k_for_375(FIXED_500_ONNX, H, W)
    
    print(f"\n[2] 修正 ONNX TopK K 值 ...")
    patch_onnx_topk(FIXED_500_ONNX, FIXED_375_ONNX, corrected_k)
    
    print(f"\n[3] trtexec 构建引擎 (预计 5-10 分钟) ...")
    build_engine(FIXED_375_ONNX, OUTPUT_ENGINE, H, W)
