"""
Export all-MiniLM-L6-v2 to ONNX format with optional INT8 quantization.

Usage:
    python training/scripts/export_onnx.py
    python training/scripts/export_onnx.py --quantize  # INT8 quantization (~22MB)
"""

import argparse
import shutil
from pathlib import Path


def export_to_onnx(output_dir: str, quantize: bool = False):
    """
    Export sentence-transformers model to ONNX format.

    Args:
        output_dir: Directory to save ONNX model
        quantize: If True, apply INT8 quantization for smaller size
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: Install required packages:")
        print("  pip install optimum[onnxruntime] onnx")
        return

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_name} to ONNX...")
    print(f"Output directory: {output_path}")

    # Export to ONNX
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
    )

    # Save model
    onnx_path = output_path / "model_fp32"
    model.save_pretrained(onnx_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(onnx_path)

    print(f"FP32 model saved to: {onnx_path}")

    # Calculate size
    model_file = onnx_path / "model.onnx"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"FP32 model size: {size_mb:.1f} MB")

    if quantize:
        print("\nApplying INT8 quantization...")

        quantized_path = output_path / "model_int8"
        quantized_path.mkdir(parents=True, exist_ok=True)

        # Configure quantization
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

        # Quantize
        quantizer = ORTQuantizer.from_pretrained(onnx_path)
        quantizer.quantize(save_dir=quantized_path, quantization_config=qconfig)

        # Copy tokenizer to quantized folder
        for f in onnx_path.glob("tokenizer*"):
            shutil.copy(f, quantized_path)
        for f in onnx_path.glob("vocab*"):
            shutil.copy(f, quantized_path)
        for f in onnx_path.glob("special_tokens*"):
            shutil.copy(f, quantized_path)

        # Calculate quantized size
        quantized_file = quantized_path / "model_quantized.onnx"
        if quantized_file.exists():
            size_mb = quantized_file.stat().st_size / (1024 * 1024)
            print(f"INT8 model saved to: {quantized_path}")
            print(f"INT8 model size: {size_mb:.1f} MB")

    print("\nDone!")
    print("\nTo use in production, set:")
    print('  INTENT_CLASSIFIER_TYPE="onnx"')
    print(f'  ONNX_MODEL_PATH="{output_path / "model_int8" if quantize else onnx_path}"')


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--output", "-o", default="training/models/onnx", help="Output directory for ONNX model")
    parser.add_argument("--quantize", "-q", action="store_true", help="Apply INT8 quantization for smaller model size")

    args = parser.parse_args()
    export_to_onnx(args.output, args.quantize)


if __name__ == "__main__":
    main()
