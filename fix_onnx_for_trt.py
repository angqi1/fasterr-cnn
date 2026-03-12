#!/usr/bin/env python3
"""
Fix ONNX model for TensorRT 8.5 compatibility.
Issues fixed:
1. /roi_heads/Reshape: [N,-1] from [N,60] -> replace with [-1, 60]
2. /roi_heads/Reshape_1: [N,-1,4] from [N,60] -> replace with [-1, 15, 4]
"""
import onnx
import numpy as np
from onnx import numpy_helper, helper

INPUT_MODEL = "fasterrcnn_nuscenes_sim.onnx"
OUTPUT_MODEL = "fasterrcnn_nuscenes_trt_fixed.onnx"

model = onnx.load(INPUT_MODEL)
graph = model.graph

# Prepend (not append) constant nodes so topological order is maintained
new_const_nodes = []

def add_const_shape_node(suffix, shape_values):
    out_name = f"__fixed_shape_{suffix}__"
    tensor = numpy_helper.from_array(
        np.array(shape_values, dtype=np.int64),
        name=out_name + "_t"
    )
    const_node = helper.make_node("Constant", inputs=[], outputs=[out_name], value=tensor)
    new_const_nodes.append(const_node)
    return out_name

FIX_MAP = {
    "/roi_heads/Reshape":   [-1, 60],
    "/roi_heads/Reshape_1": [-1, 15, 4],
}

fixed = 0
for node in graph.node:
    if node.op_type == "Reshape" and node.name in FIX_MAP:
        new_shape = FIX_MAP[node.name]
        new_shape_name = add_const_shape_node(node.name.replace("/", "_"), new_shape)
        node.input[1] = new_shape_name
        print(f"[FIX] {node.name}: replaced shape input -> {new_shape}")
        fixed += 1

# Prepend new constant nodes to maintain topological order
for cn in reversed(new_const_nodes):
    graph.node.insert(0, cn)

print(f"\nFixed {fixed} Reshape nodes")
onnx.save(model, OUTPUT_MODEL)
print(f"Saved to {OUTPUT_MODEL}")

try:
    onnx.checker.check_model(OUTPUT_MODEL)
    print("Model validation PASSED")
except Exception as e:
    print(f"Model validation warning: {e}")
