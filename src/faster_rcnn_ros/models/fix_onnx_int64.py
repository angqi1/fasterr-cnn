#!/usr/bin/env python3
"""
Fix ONNX model: convert all INT64 weights and attributes to INT32
Required for TensorRT compatibility (TRT doesn't support INT64)
"""

import onnx
from onnx import TensorProto, helper
import argparse

def convert_int64_to_int32(model_path, output_path):
    print(f"Loading {model_path}...")
    model = onnx.load(model_path)

    # 1. Fix initializers (weights)
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            print(f"Converting initializer '{init.name}' from INT64 to INT32")
            # Read raw data as int64, convert to int32
            int64_data = onnx.numpy_helper.to_array(init)
            int32_data = int64_data.astype('int32')
            # Replace with new int32 tensor
            new_init = onnx.numpy_helper.from_array(int32_data, init.name)
            init.CopyFrom(new_init)

    # 2. Fix node attributes (e.g., ConstantOfShape value attr)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                if attr.t.data_type == TensorProto.INT64:
                    print(f"Converting attribute tensor in node '{node.name}' ({node.op_type}) from INT64 to INT32")
                    int64_data = onnx.numpy_helper.to_array(attr.t)
                    int32_data = int64_data.astype('int32')
                    new_tensor = onnx.numpy_helper.from_array(int32_data)
                    attr.t.CopyFrom(new_tensor)
            elif attr.type == onnx.AttributeProto.INTS:
                # Some attributes like 'axes' might be int64 list
                if hasattr(attr, 'ints') and len(attr.ints) > 0:
                    # Check if values exceed int32 range (unlikely for axes)
                    new_ints = []
                    for val in attr.ints:
                        if val > 2**31 - 1 or val < -2**31:
                            print(f"Warning: INT64 value {val} in attribute '{attr.name}' exceeds INT32 range!")
                        new_ints.append(int(val) & 0xFFFFFFFF)  # Truncate to 32-bit
                    attr.ints[:] = new_ints

    # 3. Update opset version if needed (optional)
    # model.opset_import[0].version = 11  # Ensure compatible opset

    print(f"Saving fixed model to {output_path}...")
    onnx.save(model, output_path)
    print("✅ Fix completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model (.onnx)")
    parser.add_argument("--output", required=True, help="Output fixed ONNX model (.onnx)")
    args = parser.parse_args()
    
    convert_int64_to_int32(args.input, args.output)