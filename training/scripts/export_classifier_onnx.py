"""
Export fine-tuned BertForSequenceClassification to ONNX with INT8 quantization.

Uses torch.onnx.export + onnxruntime quantization (no optimum dependency needed).

Usage:
    python training/scripts/export_classifier_onnx.py
    python training/scripts/export_classifier_onnx.py --no-quantize
    python training/scripts/export_classifier_onnx.py --input path/to/model --output path/to/onnx
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_classifier_to_onnx(input_dir: str, output_dir: str, quantize: bool = True):
    """
    Export a fine-tuned BertForSequenceClassification model to ONNX format.

    Args:
        input_dir: Path to the fine-tuned model (model.safetensors, config.json, tokenizer)
        output_dir: Path to save the ONNX model
        quantize: If True, apply INT8 quantization (~23MB instead of ~90MB)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return

    print(f"Input model: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Quantize: {quantize}")
    print()

    # Step 1: Load model and tokenizer
    print("Step 1: Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(str(input_path))
    tokenizer = AutoTokenizer.from_pretrained(str(input_path))
    model.eval()

    num_labels = model.config.num_labels
    id2label = model.config.id2label
    print(f"  Model loaded: {num_labels} labels -> {list(id2label.values())}")

    # Step 2: Export to ONNX
    print("\nStep 2: Exporting to ONNX...")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy = tokenizer("test message", return_tensors="pt", padding=True, truncation=True, max_length=128)
    dummy_inputs = (dummy["input_ids"], dummy["attention_mask"], dummy.get("token_type_ids"))

    # Remove None inputs
    input_names = ["input_ids", "attention_mask"]
    dummy_tuple = (dummy["input_ids"], dummy["attention_mask"])
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")
        dummy_tuple = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])

    fp32_file = output_path / "model.onnx"

    # Force legacy TorchScript-based exporter (torch 2.10+ defaults to dynamo which
    # produces incomplete graphs for some models)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_tuple,
            str(fp32_file),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                **({"token_type_ids": {0: "batch", 1: "seq_len"}} if "token_type_ids" in input_names else {}),
                "logits": {0: "batch"},
            },
            opset_version=14,
            dynamo=False,
        )

    size_mb = fp32_file.stat().st_size / (1024 * 1024)
    print(f"  FP32 model exported: {size_mb:.1f} MB")

    if quantize:
        # Step 3: INT8 quantization
        print("\nStep 3: Applying INT8 quantization...")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_file = output_path / "model_quantized.onnx"
        quantize_dynamic(
            str(fp32_file),
            str(quantized_file),
            weight_type=QuantType.QInt8,
        )

        size_mb = quantized_file.stat().st_size / (1024 * 1024)
        print(f"  INT8 model: {size_mb:.1f} MB")

        # Remove FP32 model (keep only quantized)
        fp32_file.unlink()
        print("  Removed FP32 model (keeping INT8 only)")

    # Step 4: Copy tokenizer and config files
    print("\nStep 4: Copying tokenizer and config files...")
    for pattern in ["tokenizer*", "vocab*", "special_tokens*", "config.json"]:
        for f in input_path.glob(pattern):
            shutil.copy(f, output_path)
            print(f"  Copied: {f.name}")

    # Copy centroids if they exist
    for extra in ["centroids.pkl", "label_mappings.json", "centroid_metadata.json"]:
        src = input_path / extra
        if src.exists():
            shutil.copy(src, output_path)
            print(f"  Copied: {extra}")

    # Step 5: Verify the exported model
    print("\nStep 5: Verifying exported model...")
    onnx_file = output_path / "model_quantized.onnx" if quantize else output_path / "model.onnx"
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    test_input = tokenizer("What skills do I need for data science?", return_tensors="np", padding=True, truncation=True, max_length=128)
    feed = {"input_ids": test_input["input_ids"], "attention_mask": test_input["attention_mask"]}
    ort_input_names = [i.name for i in session.get_inputs()]
    if "token_type_ids" in ort_input_names:
        feed["token_type_ids"] = test_input.get("token_type_ids", np.zeros_like(test_input["input_ids"]))

    outputs = session.run(None, feed)
    logits = outputs[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    best_idx = int(np.argmax(probs))
    # id2label keys may be int or str depending on transformers version
    def get_label(idx):
        return id2label.get(idx, id2label.get(str(idx), f"label_{idx}"))

    best_label = get_label(best_idx)

    print(f"  Test: 'What skills do I need for data science?'")
    print(f"  Result: {best_label} ({probs[best_idx]:.1%})")
    print(f"  All probabilities: {dict(zip([get_label(i) for i in range(num_labels)], [f'{p:.1%}' for p in probs]))}")

    print(f"\nDone! ONNX model ready at: {output_path}")
    print(f"Files: {sorted([f.name for f in output_path.iterdir()])}")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned classifier to ONNX")
    parser.add_argument(
        "--input", "-i",
        default="training/models/all-MiniLM-L6-v2/primary/final",
        help="Input model directory",
    )
    parser.add_argument(
        "--output", "-o",
        default="training/models/onnx/classifier",
        help="Output ONNX directory",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization (keep FP32)",
    )

    args = parser.parse_args()
    export_classifier_to_onnx(args.input, args.output, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()
