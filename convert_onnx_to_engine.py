import tensorrt as trt
import argparse
import os

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    """
    Build a TensorRT engine from an ONNX file.

    Args:
        onnx_file_path (str): Path to the ONNX model file.
        engine_file_path (str): Path to save the TensorRT engine file.
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Create a builder configuration
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 if supported

        # Parse ONNX file
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"TensorRT ONNX Parser Error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX file.")

        # Build engine
        print("Building TensorRT engine. This may take a few minutes...")
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine.")

        # Save engine to file
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"TensorRT engine saved to {engine_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine.")
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model file.")
    parser.add_argument("--engine", required=True, help="Path to save the TensorRT engine file.")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    build_engine(args.onnx, args.engine)

if __name__ == "__main__":
    main()