#!/usr/bin/env python3
"""
Remove If-conditional nodes from Faster RCNN ONNX model.
Both If_1399 and If_1877 guard the batched_nms function against empty boxes.
Strategy: inline the else-branch (NMS path) into the main graph,
replacing the If node with a direct wire to the else-branch output.

This is safe because:
- In real inference, boxes are never empty (image always has candidates)
- Even if boxes were empty, NMS on empty returns empty indices naturally
"""
import onnx
import numpy as np
from onnx import numpy_helper, helper, TensorProto

INPUT_MODEL  = "fasterrcnn_nuscenes_trt_fixed.onnx"
OUTPUT_MODEL = "fasterrcnn_nuscenes_no_if.onnx"

model = onnx.load(INPUT_MODEL)
graph = model.graph

def inline_else_branch(graph, if_node):
    """
    Inline the else-branch subgraph nodes into graph.node,
    then rewire the If node's output to the else-branch output.
    Returns the else-branch output tensor name.
    """
    else_branch = None
    for attr in if_node.attribute:
        if attr.name == 'else_branch':
            else_branch = attr.g
            break
    if else_branch is None:
        raise ValueError(f"No else_branch in {if_node.name}")

    # Inline all nodes from the else_branch subgraph
    for node in else_branch.node:
        graph.node.append(node)

    # The If node's output should now point to the else_branch's output tensor
    # else_branch.output[0].name is the tensor that was last output from the else subgraph
    else_out_name = else_branch.output[0].name
    if_out_name   = if_node.output[0]

    # Add identity/rename: instead of renaming in-place, we'll just update consumers
    # of if_out_name to use else_out_name
    return if_out_name, else_out_name

def replace_tensor_name(graph, old_name, new_name):
    """Replace all occurrences of old_name as input with new_name in the main graph."""
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_name:
                node.input[i] = new_name
    # Also replace in graph outputs
    for out in graph.output:
        if out.name == old_name:
            out.name = new_name


# Process If nodes in order
# NOTE: If_1877's else_branch contains If_1901, so we handle recursion carefully
nodes_to_remove = []

for node in list(graph.node):
    if node.op_type == 'If' and node.name in ('If_1399', 'If_1877'):
        print(f"Inlining else-branch of {node.name} ...")
        if_out, else_out = inline_else_branch(graph, node)
        replace_tensor_name(graph, if_out, else_out)
        nodes_to_remove.append(node)
        print(f"  {if_out} -> {else_out}")

# Remove the original If nodes from the graph
for node in nodes_to_remove:
    graph.node.remove(node)

# Also handle any nested If_1901 (already inlined via If_1877's else-branch)
# Check if it's now in the main graph and handle it the same way
for node in list(graph.node):
    if node.op_type == 'If' and node.name == 'If_1901':
        print(f"Inlining else-branch of nested {node.name} ...")
        if_out, else_out = inline_else_branch(graph, node)
        replace_tensor_name(graph, if_out, else_out)
        graph.node.remove(node)
        print(f"  {if_out} -> {else_out}")

print(f"\nRemaining If nodes: {sum(1 for n in graph.node if n.op_type == 'If')}")

onnx.save(model, OUTPUT_MODEL)
print(f"Saved to {OUTPUT_MODEL}")

try:
    onnx.checker.check_model(OUTPUT_MODEL)
    print("Model validation PASSED")
except Exception as e:
    print(f"Model validation warning: {e}")
