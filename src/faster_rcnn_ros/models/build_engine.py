#!/usr/bin/env python3
"""
Fix fasterrcnn_nuscenes.onnx (10-class NuScenes) and build TensorRT engine.

TRT 8.5 compatibility fixes applied (6 steps):
  1. Reshape: replace dynamic concat-based shapes with static constants
  2. If nodes: inline branches to eliminate If nodes entirely
  3. TopK: replace dynamic K with static constants (for 375×1242 input)
  4. ConstantOfShape: fold Shape(ConstantOfShape(x))→x, DCE, INT64 handling
  5. INT64→INT32 conversion (skip ConstantOfShape.value to avoid byte collision)
  6. Topological sort

Usage:
  python3 build_engine.py
  # Reads:  fasterrcnn_nuscenes.onnx  (in same directory)
  # Writes: faster_rcnn_new.engine    (in same directory)
"""

import os
import sys
import copy
import subprocess
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description='Fix fasterrcnn_nuscenes.onnx and build TensorRT FP16 engine.'
    )
    p.add_argument('--height',     type=int,   default=375,  help='Input height (default 375)')
    p.add_argument('--width',      type=int,   default=1242, help='Input width  (default 1242)')
    p.add_argument('--cublas-lt',  action='store_true',
                   help='Enable cuBLASLt as extra tactic source '
                        '(may improve FC/GEMM layers in RoI Head)')
    p.add_argument('--topk-scale', type=float, default=1.0,
                   help='Scale factor for RPN TopK K values '
                        '(e.g. 0.5 halves proposals, may reduce RoI Head latency)')
    p.add_argument('--suffix',     type=str,   default='',
                   help='Optional suffix appended to engine filename '
                        '(e.g. --suffix _cublaslt → faster_rcnn_375_cublaslt.engine)')
    return p.parse_args()

_args = parse_args()
INPUT_H = _args.height
INPUT_W = _args.width

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_ONNX   = os.path.join(SCRIPT_DIR, 'fasterrcnn_nuscenes.onnx')
FIXED_ONNX   = os.path.join(SCRIPT_DIR, f'fasterrcnn_nuscenes_fixed_{INPUT_H}x{INPUT_W}.onnx')
# 引擎命名：--suffix 为空时用 faster_rcnn_{H}.engine，否则用 faster_rcnn{suffix}.engine
_suffix = _args.suffix
_engine_stem = f'faster_rcnn{_suffix}' if _suffix else f'faster_rcnn_{INPUT_H}'
OUTPUT_ENGINE = os.path.join(SCRIPT_DIR, f'{_engine_stem}.engine')
# install 目录（相对 SCRIPT_DIR 推算ros2 workspace 根目录）
# SCRIPT_DIR = .../ros2_ws/src/faster_rcnn_ros/models
# install  = .../ros2_ws/install/faster_rcnn_ros/share/faster_rcnn_ros/models
_WS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
INSTALL_DIR = os.path.join(_WS_DIR, 'install', 'faster_rcnn_ros', 'share',
                           'faster_rcnn_ros', 'models')
TRTEXEC      = '/usr/src/tensorrt/bin/trtexec'

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Fix Reshape wildcard shapes (10-class model: 40 = 10×4)
# ─────────────────────────────────────────────────────────────────────────────
RESHAPE_TARGETS = {
    '/roi_heads/Reshape':   np.array([-1, 40],    dtype=np.int64),  # bbox_pred [N,40]
    '/roi_heads/Reshape_1': np.array([-1, 10, 4], dtype=np.int64),  # [N,10cls,4coord]
}

def fix_reshape_wildcards(graph):
    new_const_nodes = []
    for node in graph.node:
        if node.op_type == 'Reshape' and node.name in RESHAPE_TARGETS:
            shape_vals = RESHAPE_TARGETS[node.name]
            const_out  = f'{node.name}_trt_shape'
            const_node = helper.make_node(
                'Constant', inputs=[], outputs=[const_out],
                name=f'{node.name}_trt_shape_node',
                value=numpy_helper.from_array(shape_vals),
            )
            new_const_nodes.append(const_node)
            node.input[1] = const_out
            print(f'  Fixed Reshape: {node.name} → {shape_vals.tolist()}')
    for n in reversed(new_const_nodes):
        graph.node.insert(0, n)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: If-node inlining
# ─────────────────────────────────────────────────────────────────────────────
def get_branch(if_node, branch_name):
    for attr in if_node.attribute:
        if attr.name == branch_name:
            return attr.g
    return None

def inline_if_node(graph_nodes_list, if_node, branch_name):
    branch = get_branch(if_node, branch_name)
    if branch is None:
        print(f'  WARNING: branch {branch_name} not found in {if_node.name}')
        return []
    if_outputs  = list(if_node.output)
    branch_outs = [o.name for o in branch.output]
    new_nodes   = [copy.deepcopy(n) for n in branch.node]
    for bout, ifout in zip(branch_outs, if_outputs):
        if bout != ifout:
            new_nodes.append(helper.make_node(
                'Identity', [bout], [ifout],
                name=f'bridge_{if_node.name}_{ifout}',
            ))
    return new_nodes

def inline_nested_if_in_subgraph(subgraph, nested_if_name, branch_name):
    for i, node in enumerate(subgraph.node):
        if node.op_type == 'If' and node.name == nested_if_name:
            branch = get_branch(node, branch_name)
            if branch is None:
                return
            if_outputs  = list(node.output)
            branch_outs = [o.name for o in branch.output]
            new_nodes   = [copy.deepcopy(n) for n in branch.node]
            for bout, ifout in zip(branch_outs, if_outputs):
                if bout != ifout:
                    new_nodes.append(helper.make_node(
                        'Identity', [bout], [ifout],
                        name=f'bridge_{nested_if_name}_{ifout}',
                    ))
            rebuilt = list(subgraph.node)
            rebuilt.pop(i)
            rebuilt.extend(new_nodes)
            del subgraph.node[:]
            subgraph.node.extend(rebuilt)
            print(f'  Inlined nested {nested_if_name} ({branch_name})')
            return

# If node inline plan (10-class NuScenes model, node names verified on ONNX):
# If_1537: RPN NMS        → else_branch  (boxes always exist in inference)
# If_2071: roi NMS        → else_branch  (detections always exist)
#   └─ If_2099 nested in If_2071 else_branch → then_branch (pre-inline first)
# box_roi_pool/If_{0-3}  → then_branch   (Squeeze 1D path)
INLINE_PLAN = {
    'If_1537':                       'else_branch',
    'If_2071':                       'else_branch',
    '/roi_heads/box_roi_pool/If':    'then_branch',
    '/roi_heads/box_roi_pool/If_1':  'then_branch',
    '/roi_heads/box_roi_pool/If_2':  'then_branch',
    '/roi_heads/box_roi_pool/If_3':  'then_branch',
}

def inline_all_if_nodes(graph):
    # Pre-inline nested If_2099 inside If_2071's else_branch
    for node in graph.node:
        if node.op_type == 'If' and node.name == 'If_2071':
            else_branch = get_branch(node, 'else_branch')
            if else_branch is not None:
                inline_nested_if_in_subgraph(else_branch, 'If_2099', 'then_branch')
            break

    nodes_to_keep = []
    extra_nodes   = []
    for node in graph.node:
        if node.op_type == 'If' and node.name in INLINE_PLAN:
            branch = INLINE_PLAN[node.name]
            print(f'  {node.name} → {branch}')
            extra_nodes.extend(inline_if_node(list(graph.node), node, branch))
        else:
            nodes_to_keep.append(node)
    del graph.node[:]
    graph.node.extend(nodes_to_keep + extra_nodes)
    print(f'  Graph now has {len(graph.node)} nodes')

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: TopK static K – computed dynamically for the target input size
# ─────────────────────────────────────────────────────────────────────────────
_TOPK_TENSOR_NAMES = [
    '/rpn/Reshape_26_output_0',
    '/rpn/Reshape_27_output_0',
    '/rpn/Reshape_28_output_0',
    '/rpn/Reshape_29_output_0',
    '/rpn/Reshape_30_output_0',
]

def compute_topk_k_map(onnx_path, input_h, input_w):
    """Run the original ONNX model at (input_h × input_w) via ORT to get the
    ReduceMin outputs that become the TopK K values for each RPN level."""
    import onnxruntime as ort
    import io as _io
    model = onnx.load(onnx_path)
    # Expose the ReduceMin outputs as graph outputs for inspection
    reduce_names = [
        '/rpn/ReduceMin_output_0',
        '/rpn/ReduceMin_1_output_0',
        '/rpn/ReduceMin_2_output_0',
        '/rpn/ReduceMin_3_output_0',
        '/rpn/ReduceMin_4_output_0',
    ]
    for n in reduce_names:
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(n, TensorProto.INT64, None)
        )
    buf = _io.BytesIO()
    onnx.save(model, buf); buf.seek(0)
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(buf.read(), sess_options=so,
                                providers=['CPUExecutionProvider'])
    fake = np.zeros((1, 3, input_h, input_w), dtype=np.float32)
    out_names = [o.name for o in sess.get_outputs()]
    inp_name = sess.get_inputs()[0].name  # 兼容 "image" 和 "input_image"
    results = sess.run(out_names[-5:], {inp_name: fake})
    k_map = {t: int(v.flat[0]) for t, v in zip(_TOPK_TENSOR_NAMES, results)}
    return k_map

# Populated in main() after INPUT_H/INPUT_W are known
TOPK_K_MAP: dict = {}

def fold_topk_k_to_constants(graph):
    count = 0
    for k_tensor, k_value in TOPK_K_MAP.items():
        init_name = f'topk_k_const_{k_tensor.replace("/", "_")}'
        graph.initializer.append(
            numpy_helper.from_array(np.array([k_value], dtype=np.int64), name=init_name)
        )
        for node in graph.node:
            if node.op_type == 'TopK' and len(node.input) > 1 and node.input[1] == k_tensor:
                node.input[1] = init_name
                count += 1
    print(f'  Folded {count} TopK K tensors to static constants')

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: ConstantOfShape fix (three sub-steps)
#   A) Fold:  Shape(ConstantOfShape(x)) → Identity(x)
#   B) Elim:  Add(ConstantOfShape(0), x) / Add(x, ConstantOfShape(0)) → Identity(x)
#   C) DCE:   remove dead ConstantOfShape nodes
#
# Background: TRT 8.5 hashes weights by raw bytes. INT32(0) and FLOAT32(0.0)
# both produce 4 bytes \x00\x00\x00\x00, causing a collision. Steps A+B remove
# all zero-ConstantOfShape usages so DCE can delete them all. Step 5 then keeps
# the remaining ConstantOfShape.value as INT64 (8 bytes) to avoid collision.
# ─────────────────────────────────────────────────────────────────────────────
def fold_shape_of_constantofshape(graph):
    """Shape(ConstantOfShape(x)) → Identity(x)  (shape of output = x)"""
    cos_out_to_input = {}
    for node in graph.node:
        if node.op_type == 'ConstantOfShape' and node.input and node.output:
            cos_out_to_input[node.output[0]] = node.input[0]

    count = 0
    for node in graph.node:
        if node.op_type == 'Shape' and node.input[0] in cos_out_to_input:
            orig_input = cos_out_to_input[node.input[0]]
            node.op_type = 'Identity'
            del node.input[:]
            node.input.append(orig_input)
            count += 1
    print(f'  Folded {count} Shape(ConstantOfShape(x)) → Identity(x)')

def eliminate_add_zero_patterns(graph):
    """Add(ConstantOfShape(0), x) / Add(x, ConstantOfShape(0)) → Identity(x)"""
    # Collect ConstantOfShape outputs that produce all-zero tensors
    cos_zero_outputs = set()
    for node in graph.node:
        if node.op_type == 'ConstantOfShape':
            for attr in node.attribute:
                if attr.name == 'value' and attr.HasField('t'):
                    val = numpy_helper.to_array(attr.t)
                    if val.size == 0 or (val == 0).all():
                        for o in node.output:
                            cos_zero_outputs.add(o)

    nodes_to_remove = set()   # Add node output names to remove
    nodes_to_add    = []
    for node in graph.node:
        if node.op_type == 'Add' and len(node.input) == 2:
            a, b   = node.input
            out    = node.output[0]
            keep   = None
            if a in cos_zero_outputs:
                keep = b
            elif b in cos_zero_outputs:
                keep = a
            if keep is not None:
                nodes_to_add.append(
                    helper.make_node('Identity', [keep], [out], name=f'elim_add0_{out}')
                )
                nodes_to_remove.add(out)

    if nodes_to_add:
        new_nodes = [n for n in graph.node
                     if not (n.op_type == 'Add' and n.output and n.output[0] in nodes_to_remove)]
        new_nodes.extend(nodes_to_add)
        del graph.node[:]
        graph.node.extend(new_nodes)
        print(f'  Eliminated {len(nodes_to_add)} Add-zero patterns → Identity')

def eliminate_dead_constantofshape(graph):
    """Remove ConstantOfShape nodes whose outputs are never consumed."""
    consumed = set()
    for node in graph.node:
        for inp in node.input:
            consumed.add(inp)
    for out in graph.output:
        consumed.add(out.name)

    before = len(graph.node)
    alive  = [n for n in graph.node
              if not (n.op_type == 'ConstantOfShape' and
                      all(o not in consumed for o in n.output))]
    removed = before - len(alive)
    del graph.node[:]
    graph.node.extend(alive)
    print(f'  DCE: removed {removed} dead ConstantOfShape nodes')

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: INT64 → INT32  (SKIP ConstantOfShape.value to avoid byte collision)
#
# The byte-collision: TRT hashes weights by raw bytes.
# ConstantOfShape(dtype=INT32, value=0) → 4 bytes \x00\x00\x00\x00
# ConstantOfShape(dtype=FLOAT32, value=0.0) → 4 bytes \x00\x00\x00\x00  ← same!
# Fix: keep ConstantOfShape.value as INT64 (8 bytes) so it no longer collides.
# ─────────────────────────────────────────────────────────────────────────────
def convert_int64_to_int32(model):
    INT64     = TensorProto.INT64
    INT32     = TensorProto.INT32
    INT32_MAX = np.int64(2**31 - 1)
    INT32_MIN = np.int64(-(2**31))
    count = 0

    def clamp32(arr):
        return np.clip(arr.astype(np.int64), INT32_MIN, INT32_MAX).astype(np.int32)

    # Initializers
    for init in model.graph.initializer:
        if init.data_type == INT64:
            new_init = numpy_helper.from_array(clamp32(numpy_helper.to_array(init)), name=init.name)
            init.CopyFrom(new_init)
            count += 1

    def fix_nodes(graph):
        nonlocal count
        for node in graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.HasField('t') and attr.t.data_type == INT64:
                        new_t = numpy_helper.from_array(clamp32(numpy_helper.to_array(attr.t)))
                        attr.t.CopyFrom(new_t)
                        count += 1
            elif node.op_type == 'ConstantOfShape':
                # *** SKIP value attr – preserving INT64 (8 bytes) avoids collision
                #     with FLOAT32(0.0) whose raw bytes are identical as INT32 ***
                pass
            elif node.op_type == 'Cast':
                for attr in node.attribute:
                    if attr.name == 'to' and attr.i == INT64:
                        attr.i = INT32
                        count += 1

    fix_nodes(model.graph)
    print(f'  Converted {count} INT64 references to INT32 (ConstantOfShape.value skipped)')

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Topological sort
# ─────────────────────────────────────────────────────────────────────────────
def topological_sort(graph):
    defined = set()
    for init in graph.initializer:
        defined.add(init.name)
    for inp in graph.input:
        defined.add(inp.name)

    remaining    = list(graph.node)
    sorted_nodes = []
    for _ in range(len(remaining) + 10):
        if not remaining:
            break
        progress, still_waiting = False, []
        for node in remaining:
            if all(inp == '' or inp in defined for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    if out:
                        defined.add(out)
                progress = True
            else:
                still_waiting.append(node)
        remaining = still_waiting
        if not progress:
            break

    if remaining:
        print(f'  WARNING: {len(remaining)} nodes unresolvable:')
        for n in remaining[:5]:
            print(f'    {n.op_type} {n.name}: missing {[i for i in n.input if i and i not in defined][:3]}')
        sorted_nodes.extend(remaining)

    del graph.node[:]
    graph.node.extend(sorted_nodes)
    print(f'  Topological sort done: {len(sorted_nodes)} nodes')

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_ONNX):
        print(f'ERROR: {INPUT_ONNX} not found')
        sys.exit(1)

    print(f'Loading {INPUT_ONNX} …')
    model = onnx.load(INPUT_ONNX)
    graph = model.graph
    print(f'  Nodes: {len(graph.node)}, Initializers: {len(graph.initializer)}')

    print('\n[1] Fixing Reshape wildcard shapes …')
    fix_reshape_wildcards(graph)

    print('\n[2] Inlining If nodes …')
    inline_all_if_nodes(graph)

    print('\n[3] Topological sort (after If inlining) …')
    topological_sort(graph)

    print(f'\n[4] Computing TopK K values for {INPUT_H}×{INPUT_W} via ORT …')
    global TOPK_K_MAP
    TOPK_K_MAP = compute_topk_k_map(INPUT_ONNX, INPUT_H, INPUT_W)
    if _args.topk_scale != 1.0:
        TOPK_K_MAP = {k: max(1, int(v * _args.topk_scale)) for k, v in TOPK_K_MAP.items()}
        print(f'  [TopK scale={_args.topk_scale}] Scaled K values:')
    for t, v in TOPK_K_MAP.items():
        print(f'  {t.split("/")[-1]} = {v}')
    print(f'  Total pre-NMS proposals (sum): {sum(TOPK_K_MAP.values())}')

    print('\n[4b] TopK K → static constants …')
    fold_topk_k_to_constants(graph)

    print('\n[5a] Folding Shape(ConstantOfShape(x)) → Identity(x) …')
    fold_shape_of_constantofshape(graph)

    print('\n[5b] Eliminating Add(ConstantOfShape(0), x) → Identity(x) …')
    eliminate_add_zero_patterns(graph)

    print('\n[5c] Dead code elimination (ConstantOfShape) …')
    eliminate_dead_constantofshape(graph)

    print('\n[6] INT64 → INT32 (ConstantOfShape.value kept as INT64) …')
    convert_int64_to_int32(model)

    print('\n[7] Final topological sort …')
    topological_sort(graph)

    # Rename input_image → image for TRT compatibility
    for inp in graph.input:
        if inp.name == 'input_image':
            old_name = inp.name
            inp.name = 'image'
            for node in graph.node:
                node.input[:] = ['image' if x == old_name else x for x in node.input]
            print("  Renamed input 'input_image' → 'image'")

    print(f'\n[8] Saving fixed ONNX to {FIXED_ONNX} …')
    onnx.save(model, FIXED_ONNX)
    print(f'  Saved ({len(graph.node)} nodes)')

    print('\n[9] ONNX check …')
    try:
        onnx.checker.check_model(FIXED_ONNX)
        print('  PASSED')
    except Exception as e:
        print(f'  WARNING: {e}')

    print(f'\n[10] Building TRT engine with trtexec …')
    cmd = [
        TRTEXEC,
        f'--onnx={FIXED_ONNX}',
        f'--saveEngine={OUTPUT_ENGINE}',
        '--fp16',
        f'--minShapes=image:1x3x{INPUT_H}x{INPUT_W}',
        f'--optShapes=image:1x3x{INPUT_H}x{INPUT_W}',
        f'--maxShapes=image:1x3x{INPUT_H}x{INPUT_W}',
        '--workspace=4096',
    ]
    if _args.cublas_lt:
        cmd.append('--tacticSources=+CUBLAS_LT')
        print('  [cuBLASLt] 已启用 cuBLASLt 作为 tactic source（对 RoI Head FC 层有潜在收益）')
    print(' ', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    tail = lambda s: s[-4000:] if len(s) > 4000 else s
    if result.stdout:
        print('\n=== trtexec STDOUT ===\n' + tail(result.stdout))
    if result.stderr:
        print('\n=== trtexec STDERR ===\n' + tail(result.stderr))

    if result.returncode == 0:
        size_mb = os.path.getsize(OUTPUT_ENGINE) / 1024 / 1024
        print(f'\n✅ SUCCESS! Engine: {OUTPUT_ENGINE} ({size_mb:.0f} MB)')
        # 自动拷贝到 install/ 供 bench_single.py 和 ROS2 节点使用
        install_dir = os.path.normpath(INSTALL_DIR)
        if os.path.isdir(install_dir):
            import shutil
            dst = os.path.join(install_dir, os.path.basename(OUTPUT_ENGINE))
            shutil.copy2(OUTPUT_ENGINE, dst)
            print(f'  📦 已拷贝到: {dst}')
        else:
            print(f'  ⚠️  install 目录不存在，跳过拷贝: {install_dir}')
    else:
        print(f'\n❌ trtexec failed (returncode={result.returncode})')
        sys.exit(1)

if __name__ == '__main__':
    main()
