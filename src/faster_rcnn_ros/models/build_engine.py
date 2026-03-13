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

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_ONNX   = os.path.join(SCRIPT_DIR, 'fasterrcnn_nuscenes.onnx')
FIXED_ONNX   = os.path.join(SCRIPT_DIR, 'fasterrcnn_nuscenes_fixed_new.onnx')
OUTPUT_ENGINE = os.path.join(SCRIPT_DIR, 'faster_rcnn_new.engine')
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
# Step 3: TopK static K (precomputed for 375×1242 input)
# ─────────────────────────────────────────────────────────────────────────────
TOPK_K_MAP = {
    '/rpn/Reshape_26_output_0': 1000,
    '/rpn/Reshape_27_output_0': 1000,
    '/rpn/Reshape_28_output_0': 1000,
    '/rpn/Reshape_29_output_0': 420,
    '/rpn/Reshape_30_output_0': 120,
}

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
# Step 4: ConstantOfShape fix
#   A) Fold:  Shape(ConstantOfShape(x)) → x
#   B) DCE:   remove dead ConstantOfShape nodes
# ─────────────────────────────────────────────────────────────────────────────
def fold_shape_of_constantofshape(graph):
    """Shape(ConstantOfShape(x)) → x  (result equals original shape input x)"""
    cos_out_to_input = {}
    for node in graph.node:
        if node.op_type == 'ConstantOfShape' and node.input and node.output:
            cos_out_to_input[node.output[0]] = node.input[0]

    count = 0
    for node in graph.node:
        if node.op_type == 'Shape' and node.input[0] in cos_out_to_input:
            # Replace: Shape(cos_out) → Identity(original_input_of_cos)
            orig_input = cos_out_to_input[node.input[0]]
            node.op_type = 'Identity'
            del node.input[:]
            node.input.append(orig_input)
            count += 1
    print(f'  Folded {count} Shape(ConstantOfShape(x)) → x')

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

    print('\n[4] TopK K → static constants …')
    fold_topk_k_to_constants(graph)

    print('\n[5a] Folding Shape(ConstantOfShape(x)) → x …')
    fold_shape_of_constantofshape(graph)

    print('\n[5b] Dead code elimination (ConstantOfShape) …')
    eliminate_dead_constantofshape(graph)

    print('\n[6] INT64 → INT32 (ConstantOfShape.value kept as INT64) …')
    convert_int64_to_int32(model)

    print('\n[7] Final topological sort …')
    topological_sort(graph)

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
        '--minShapes=image:1x3x375x1242',
        '--optShapes=image:1x3x375x1242',
        '--maxShapes=image:1x3x375x1242',
        '--workspace=4096',
    ]
    print(' ', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    tail = lambda s: s[-4000:] if len(s) > 4000 else s
    if result.stdout:
        print('\n=== trtexec STDOUT ===\n' + tail(result.stdout))
    if result.stderr:
        print('\n=== trtexec STDERR ===\n' + tail(result.stderr))

    if result.returncode == 0:
        size_mb = os.path.getsize(OUTPUT_ENGINE) / 1024 / 1024
        print(f'\n✅ SUCCESS! Engine: {OUTPUT_ENGINE} ({size_mb:.0f} MB)')
    else:
        print(f'\n❌ trtexec failed (returncode={result.returncode})')
        sys.exit(1)

if __name__ == '__main__':
    main()


# Set logging severity
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path, input_shape=(1, 3, 800, 1344), fp16=True):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_file_path (str): Path to input ONNX model
        engine_file_path (str): Path to output .engine file
        input_shape (tuple): Input shape (N, C, H, W)
        fp16 (bool): Enable FP16 precision (recommended for Orin)
    """
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

    # Create builder and network
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 2 * (1 << 30)  # 2 GB

        # Enable FP16 if supported
        if fp16 and builder.platform_has_fast_fp16:
            print("Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)

        # Parse ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse ONNX file")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        # Set input shape
        input_tensor = network.get_input(0)
        print(f"Input name: {input_tensor.name}, shape: {input_shape}")
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, min=input_shape, opt=input_shape, max=input_shape)
        config.add_optimization_profile(profile)

        # Build serialized engine
        print("Building TensorRT engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"✅ Engine saved to: {engine_file_path}")
        return engine_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model")
    parser.add_argument("--engine", required=True, help="Path to output .engine file")
    parser.add_argument("--height", type=int, default=800, help="Input height (default: 800)")
    parser.add_argument("--width", type=int, default=1344, help="Input width (default: 1344)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 (default: enabled if supported)")

    args = parser.parse_args()

    input_shape = (1, 3, args.height, args.width)
    build_engine(args.onnx, args.engine, input_shape=input_shape, fp16=args.fp16)