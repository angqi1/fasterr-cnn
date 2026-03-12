"""
Build TensorRT engine from ONNX for Jetson AGX Orin (JetPack 5.1.2, TRT 8.5.2)
"""

import os
import argparse
import tensorrt as trt

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