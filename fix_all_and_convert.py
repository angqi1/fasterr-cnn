#!/usr/bin/env python3
"""
Comprehensive fix for fasterrcnn_nuscenes.onnx → TensorRT engine.

Strategy:
1. Start from ORIGINAL fasterrcnn_nuscenes.onnx (NOT onnxsim version)
2. Fix Reshape wildcard nodes ([0,-1] → static shapes)
3. Inline all If nodes:
   - If_1399 (RPN NMS):           take else_branch
   - If_1877 (roi_heads NMS):     take else_branch
   - If_1901 (nested, always True): take then_branch (Squeeze)
   - box_roi_pool/If, If_1, If_2, If_3: take else_branch (Identity pass-through)
4. Topological sort
5. Save as fasterrcnn_nuscenes_fixed_final.onnx
6. Convert with trtexec
"""

import onnx
import numpy as np
from onnx import numpy_helper, helper, TensorProto
import copy
import subprocess
import sys


# ---------------------------------------------------------------------------
# Step 1: Fix Reshape wildcards
# ---------------------------------------------------------------------------

def fix_reshape_wildcards(graph):
    """Replace dynamic concat-based shape inputs with static constants."""
    reshape_target_shapes = {
        '/roi_heads/Reshape':   np.array([-1, 60],     dtype=np.int64),
        '/roi_heads/Reshape_1': np.array([-1, 15, 4],  dtype=np.int64),
    }

    new_const_nodes = []
    for node in graph.node:
        if node.op_type == 'Reshape' and node.name in reshape_target_shapes:
            shape_vals = reshape_target_shapes[node.name]
            const_out  = f'{node.name}_trt_shape'
            const_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=[const_out],
                name=f'{node.name}_trt_shape_node',
                value=numpy_helper.from_array(shape_vals),
            )
            new_const_nodes.append(const_node)
            node.input[1] = const_out
            print(f'  Fixed Reshape: {node.name} → shape {shape_vals.tolist()}')

    # Prepend constants so they appear before the Reshape nodes
    for n in reversed(new_const_nodes):
        graph.node.insert(0, n)


# ---------------------------------------------------------------------------
# Step 2: If-node inlining helpers
# ---------------------------------------------------------------------------

def get_branch(if_node, branch_name):
    for attr in if_node.attribute:
        if attr.name == branch_name:
            return attr.g
    return None


def inline_if_node_into_graph(graph_nodes_list, if_node, branch_name):
    """
    Remove if_node from graph_nodes_list.
    Append nodes from the chosen branch + Identity mapping nodes.
    Returns the list of new nodes added.
    """
    branch = get_branch(if_node, branch_name)
    if branch is None:
        print(f'  WARNING: branch {branch_name} not found in {if_node.name}')
        return []

    if_outputs    = list(if_node.output)
    branch_outs   = [o.name for o in branch.output]

    new_nodes = [copy.deepcopy(n) for n in branch.node]

    # Add Identity nodes to bridge branch outputs → if node outputs
    for bout, ifout in zip(branch_outs, if_outputs):
        if bout != ifout:
            id_node = helper.make_node(
                'Identity',
                inputs=[bout],
                outputs=[ifout],
                name=f'bridge_{if_node.name}_{ifout}',
            )
            new_nodes.append(id_node)

    return new_nodes


def inline_nested_if_in_subgraph(subgraph, nested_if_name, branch_name):
    """
    Inside 'subgraph', find nested If node named 'nested_if_name' and inline
    'branch_name' of it directly.
    """
    target_idx = None
    target_node = None
    for i, node in enumerate(subgraph.node):
        if node.op_type == 'If' and node.name == nested_if_name:
            target_idx = i
            target_node = node
            break

    if target_node is None:
        print(f'  WARNING: nested If {nested_if_name} not found in subgraph')
        return

    branch = get_branch(target_node, branch_name)
    if branch is None:
        return

    if_outputs  = list(target_node.output)
    branch_outs = [o.name for o in branch.output]

    new_nodes = [copy.deepcopy(n) for n in branch.node]
    for bout, ifout in zip(branch_outs, if_outputs):
        if bout != ifout:
            id_node = helper.make_node(
                'Identity',
                inputs=[bout],
                outputs=[ifout],
                name=f'bridge_{nested_if_name}_{ifout}',
            )
            new_nodes.append(id_node)

    # Rebuild subgraph node list
    rebuilt = list(subgraph.node)
    rebuilt.pop(target_idx)
    rebuilt.extend(new_nodes)
    del subgraph.node[:]
    subgraph.node.extend(rebuilt)
    print(f'  Inlined nested {nested_if_name} ({branch_name}) into parent subgraph')


# ---------------------------------------------------------------------------
# Step 3: Topological sort
# ---------------------------------------------------------------------------

def topological_sort(graph):
    defined = set()
    for init in graph.initializer:
        defined.add(init.name)
    for inp in graph.input:
        defined.add(inp.name)

    remaining = list(graph.node)
    sorted_nodes = []
    max_passes = len(remaining) + 10

    for _ in range(max_passes):
        if not remaining:
            break
        progress = False
        still_waiting = []
        for node in remaining:
            inputs_ready = all(
                (inp == '' or inp in defined)
                for inp in node.input
            )
            if inputs_ready:
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
        print(f'  WARNING: {len(remaining)} nodes unresolvable after sort:')
        for n in remaining[:10]:
            missing = [inp for inp in n.input if inp and inp not in defined]
            print(f'    {n.op_type} {n.name}: missing {missing[:4]}')
        sorted_nodes.extend(remaining)

    del graph.node[:]
    graph.node.extend(sorted_nodes)
    print(f'  Sorted: {len(sorted_nodes)} nodes total, {len(remaining)} warnings')


# ---------------------------------------------------------------------------
# Step 4b: Fold TopK dynamic K → static initializers
# TRT 8.5 requires TopK K as a constant/initializer, not a computed tensor.
# We substitute the precomputed K values for fixed input shape [1,3,480,640].
# ---------------------------------------------------------------------------

def fold_topk_k_to_constants(graph):
    """
    Replace dynamic TopK K tensors with constant initializers.
    Values computed by running onnxruntime on [1,3,480,640].
    """
    # Precomputed for input [1, 3, 480, 640]
    topk_k_map = {
        '/rpn/Reshape_26_output_0': 1000,
        '/rpn/Reshape_27_output_0': 1000,
        '/rpn/Reshape_28_output_0': 1000,
        '/rpn/Reshape_29_output_0': 1000,
        '/rpn/Reshape_30_output_0': 663,
    }
    count = 0
    for k_tensor, k_value in topk_k_map.items():
        # Replace the tensor with a constant initializer
        init_name = f'topk_k_const_{k_tensor.replace("/", "_")}'
        init = numpy_helper.from_array(
            np.array([k_value], dtype=np.int64), name=init_name
        )
        graph.initializer.append(init)
        # Redirect all TopK nodes using this K tensor to new initializer
        for node in graph.node:
            if node.op_type == 'TopK' and len(node.input) > 1 and node.input[1] == k_tensor:
                node.input[1] = init_name
                count += 1
    print(f'  Folded {count} TopK K tensors to static constants')


# ---------------------------------------------------------------------------
# Step 5: Eliminate Add-with-zero patterns and dead code
# (ConstantOfShape(0) + Add produces bytes \x00\x00\x00\x00 which conflicts
#  with FLOAT32(0.0) constants in TRT 8.5's weight deduplication)
# ---------------------------------------------------------------------------

def eliminate_add_zero_and_dead_code(graph):
    """
    1. Find Add nodes where one input is all-zeros ConstantOfShape.
       Replace Add(zero, x) → Identity(x).
    2. Dead code elimination: remove nodes whose outputs are never consumed.
    """
    # --- Pass 1: collect ConstantOfShape outputs that are all-zero fills ---
    cos_zero_out = set()
    for node in graph.node:
        if node.op_type == 'ConstantOfShape':
            for attr in node.attribute:
                if attr.name == 'value' and attr.HasField('t'):
                    val = numpy_helper.to_array(attr.t)
                    if val.size == 0 or (val == 0).all():
                        for o in node.output:
                            cos_zero_out.add(o)

    # --- Pass 2: replace Add(zero, x) → Identity(x) ---
    nodes_to_remove_names = set()   # track by output tensor name (not id)
    nodes_to_add    = []
    for node in graph.node:
        if node.op_type == 'Add' and len(node.input) == 2:
            a, b = node.input
            out  = node.output[0]
            replacement = None
            if a in cos_zero_out:
                replacement = b
            elif b in cos_zero_out:
                replacement = a
            if replacement is not None:
                nodes_to_add.append(
                    helper.make_node('Identity', [replacement], [out],
                                     name=f'elim_zero_add_{out}')
                )
                nodes_to_remove_names.add(out)  # use output name as key

    if nodes_to_add:
        # Remove old Add nodes (identified by output name) and add Identity replacements
        nodes_list = [n for n in graph.node
                      if not (n.op_type == 'Add' and n.output and n.output[0] in nodes_to_remove_names)]
        nodes_list.extend(nodes_to_add)
        del graph.node[:]
        graph.node.extend(nodes_list)
        print(f'  Replaced {len(nodes_to_add)} Add-zero nodes with Identity')

    # --- Pass 3: BFS-based dead code elimination ---
    # Build output→node map  (use output tensor name as key)
    out_to_node_idx = {}
    for i, node in enumerate(graph.node):
        for o in node.output:
            if o:
                out_to_node_idx[o] = i

    # Start from graph outputs and work backward
    needed_tensors = set()
    for go in graph.output:
        if go.name:
            needed_tensors.add(go.name)

    frontier = list(needed_tensors)
    while frontier:
        t = frontier.pop()
        if t not in out_to_node_idx:
            continue   # produced by initializer or graph input
        idx = out_to_node_idx[t]
        node = graph.node[idx]
        # Mark all outputs of this node as needed
        for o in node.output:
            if o and o not in needed_tensors:
                needed_tensors.add(o)
                frontier.append(o)
        # Add all inputs of this node to frontier
        for inp in node.input:
            if inp and inp not in needed_tensors:
                needed_tensors.add(inp)
                frontier.append(inp)

    # Keep nodes that produce at least one needed tensor
    before = len(graph.node)
    alive  = [n for n in graph.node if any(o in needed_tensors for o in n.output if o)]
    dead_count = before - len(alive)
    del graph.node[:]
    graph.node.extend(alive)
    print(f'  Dead code elimination: removed {dead_count} nodes, kept {len(alive)}')


# ---------------------------------------------------------------------------
# Step 6: Convert INT64 → INT32 (TRT 8.5 doesn't support INT64)
# ---------------------------------------------------------------------------

def convert_int64_to_int32(model):
    """
    Convert all INT64 constants/initializers to INT32.
    This prevents TRT's "weights of same values but different types" error
    that occurs when TRT auto-casts INT64 → INT32 internally.
    Large values (e.g. INT64_MAX used as NMS unlimited limit) are clamped to INT32_MAX.
    """
    INT64     = TensorProto.INT64
    INT32     = TensorProto.INT32
    INT32_MAX = np.int64(2**31 - 1)
    INT32_MIN = np.int64(-(2**31))
    count = 0

    def clamp_to_int32(arr_int64):
        return np.clip(arr_int64, INT32_MIN, INT32_MAX).astype(np.int32)

    # --- Initializers ---
    for init in model.graph.initializer:
        if init.data_type == INT64:
            arr = numpy_helper.to_array(init).astype(np.int64)
            new_init = numpy_helper.from_array(clamp_to_int32(arr), name=init.name)
            init.CopyFrom(new_init)
            count += 1

    # --- Constant / ConstantOfShape nodes in graph ---
    def fix_constants_in_graph(graph):
        nonlocal count
        for node in graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.HasField('t') and attr.t.data_type == INT64:
                        arr = numpy_helper.to_array(attr.t).astype(np.int64)
                        new_t = numpy_helper.from_array(clamp_to_int32(arr))
                        attr.t.CopyFrom(new_t)
                        count += 1
            elif node.op_type == 'ConstantOfShape':
                for attr in node.attribute:
                    if attr.name == 'value' and attr.HasField('t') and attr.t.data_type == INT64:
                        arr = numpy_helper.to_array(attr.t).astype(np.int64)
                        new_t = numpy_helper.from_array(clamp_to_int32(arr))
                        attr.t.CopyFrom(new_t)
                        count += 1
            # Also fix Cast nodes that cast TO INT64
            elif node.op_type == 'Cast':
                for attr in node.attribute:
                    if attr.name == 'to' and attr.i == INT64:
                        attr.i = INT32
                        count += 1

    fix_constants_in_graph(model.graph)
    print(f'  Converted {count} INT64 tensor/cast references to INT32')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    src = 'fasterrcnn_nuscenes.onnx'
    dst = 'fasterrcnn_nuscenes_fixed_final.onnx'
    engine_dst = 'faster_rcnn.engine'

    print(f'Loading {src} …')
    model = onnx.load(src)
    graph = model.graph
    print(f'  Nodes: {len(graph.node)}, Initializers: {len(graph.initializer)}')

    # ---------- Fix Reshape wildcards ----------
    print('\n[1] Fixing Reshape wildcard nodes …')
    fix_reshape_wildcards(graph)

    # ---------- Pre-process nested If_1901 ----------
    print('\n[2] Pre-processing If_1877: inlining nested If_1901 (then_branch) …')
    for node in graph.node:
        if node.op_type == 'If' and node.name == 'If_1877':
            else_branch = get_branch(node, 'else_branch')
            if else_branch is not None:
                inline_nested_if_in_subgraph(else_branch, 'If_1901', 'then_branch')
            break

    # ---------- Inline top-level If nodes ----------
    # Decision table:
    #   If_1399                       → else_branch  (RPN NMS, assume boxes exist)
    #   If_1877                       → else_branch  (roi NMS, assume detections)
    #   /roi_heads/box_roi_pool/If    → else_branch  (Identity, no squeeze needed)
    #   /roi_heads/box_roi_pool/If_1  → else_branch
    #   /roi_heads/box_roi_pool/If_2  → else_branch
    #   /roi_heads/box_roi_pool/If_3  → else_branch
    inline_plan = {
        'If_1399':                        'else_branch',
        'If_1877':                        'else_branch',
        '/roi_heads/box_roi_pool/If':     'then_branch',   # then=Squeeze (1D), else=Identity (2D)
        '/roi_heads/box_roi_pool/If_1':   'then_branch',
        '/roi_heads/box_roi_pool/If_2':   'then_branch',
        '/roi_heads/box_roi_pool/If_3':   'then_branch',
    }

    print('\n[3] Inlining top-level If nodes …')
    nodes_list     = list(graph.node)
    nodes_to_keep  = []
    extra_nodes    = []

    for node in nodes_list:
        if node.op_type == 'If' and node.name in inline_plan:
            branch = inline_plan[node.name]
            print(f'  {node.name} → {branch}')
            added = inline_if_node_into_graph(nodes_list, node, branch)
            extra_nodes.extend(added)
            # Do NOT keep original If node
        else:
            nodes_to_keep.append(node)

    new_graph_nodes = nodes_to_keep + extra_nodes
    del graph.node[:]
    graph.node.extend(new_graph_nodes)
    print(f'  Graph now has {len(graph.node)} nodes')

    # ---------- Topological sort ----------
    print('\n[4] Topological sort …')
    topological_sort(graph)

    # ---------- Eliminate Add-zero patterns + dead code ----------
    print('\n[5] Eliminating Add-zero patterns and dead code …')
    eliminate_add_zero_and_dead_code(graph)

    # ---------- Fold TopK K → static constants ----------
    print('\n[5b] Folding TopK K to static constants …')
    fold_topk_k_to_constants(graph)

    # ---------- INT64 → INT32 conversion ----------
    print('\n[6] Converting INT64 → INT32 …')
    convert_int64_to_int32(model)

    # ---------- Re-sort after all modifications ----------
    print('\n[7] Re-sorting after all modifications …')
    topological_sort(graph)

    # ---------- Save ----------
    print(f'\n[8] Saving to {dst} …')
    onnx.save(model, dst)
    print(f'  Saved ({len(graph.node)} nodes)')

    # ---------- Quick ONNX check ----------
    print('\n[9] ONNX model check …')
    try:
        onnx.checker.check_model(dst)
        print('  check_model: PASSED')
    except Exception as e:
        print(f'  check_model warning: {e}')

    # ---------- TensorRT conversion ----------
    trtexec = '/usr/src/tensorrt/bin/trtexec'
    print(f'\n[10] Converting to TRT engine with {trtexec} …')
    cmd = [
        trtexec,
        f'--onnx={dst}',
        f'--saveEngine={engine_dst}',
        '--fp16',
        '--workspace=4096',
    ]
    print('  CMD:', ' '.join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Show last 4000 chars of each stream
    def tail(s, n=4000):
        return s[-n:] if len(s) > n else s

    if result.stdout:
        print('\n=== trtexec STDOUT ===')
        print(tail(result.stdout))
    if result.stderr:
        print('\n=== trtexec STDERR ===')
        print(tail(result.stderr))

    if result.returncode == 0:
        print(f'\n✓ SUCCESS! Engine saved: {engine_dst}')
        # Copy to ROS install dir
        install_path = (
            '/home/nvidia/ros2_ws/src/faster_rcnn_ros/install/'
            'faster_rcnn_ros/share/faster_rcnn_ros/models/faster_rcnn.engine'
        )
        import shutil, os
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        shutil.copy(engine_dst, install_path)
        print(f'  Copied to {install_path}')
    else:
        print(f'\n✗ trtexec FAILED (code {result.returncode})')
        sys.exit(1)


if __name__ == '__main__':
    main()
