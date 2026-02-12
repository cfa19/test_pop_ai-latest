"""
Export all hierarchical classifier models (primary + secondary) to ONNX format.

Converts:
  - Primary model (BertForSequenceClassification / all-MiniLM-L6-v2, 8 categories)
  - 6 Secondary models (DistilBertForSequenceClassification / distilbert-base-uncased)

Each model is exported to ONNX FP32, then quantized to INT8 (~3-4x smaller).

Usage:
    python -m training.scripts.export_hierarchy_onnx

Output structure:
    training/models/full_onnx/
    ├── hierarchy_metadata.json
    ├── secondary_metadata.json
    ├── primary/
    │   ├── model.onnx
    │   ├── model_quantized.onnx
    │   ├── config.json
    │   ├── label_mappings.json
    │   ├── centroids.pkl
    │   ├── centroid_metadata.json
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    └── secondary/
        ├── aspirational/
        ├── emotional/
        ├── learning/
        ├── professional/
        ├── psychological/
        └── social/
"""

import json
import shutil
import sys
from pathlib import Path

# Add D:\pylib to path (onnx installed there to avoid Windows long path issue)
PYLIB = r"D:\pylib"
if PYLIB not in sys.path:
    sys.path.insert(0, PYLIB)

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths
HIERARCHY_DIR = Path("training/models/hierarchy")
OUTPUT_DIR = Path("training/models/full_onnx")

# Files to copy alongside each ONNX model
COPY_FILES = [
    "config.json",
    "label_mappings.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "centroids.pkl",
    "centroid_metadata.json",
]


def export_model_to_onnx(model_dir: Path, output_dir: Path, model_name: str) -> bool:
    """Export a single model to ONNX with INT8 quantization."""
    print(f"\n{'='*60}")
    print(f"Exporting: {model_name}")
    print(f"  Source:  {model_dir}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")

    if not model_dir.exists():
        print(f"  [ERROR] Source directory not found: {model_dir}")
        return False

    safetensors_file = model_dir / "model.safetensors"
    if not safetensors_file.exists():
        print(f"  [ERROR] model.safetensors not found in {model_dir}")
        return False

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model_type = config.get("model_type", "unknown")
    architectures = config.get("architectures", [])
    id2label = config.get("id2label", {})
    num_labels = len(id2label)
    print(f"  Model type: {model_type} ({architectures[0] if architectures else 'unknown'})")
    print(f"  Labels ({num_labels}): {list(id2label.values())}")

    # Load model and tokenizer
    print("  Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dummy input
    dummy_text = "This is a test message for export"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Bert uses token_type_ids, DistilBert does not
    if model_type == "distilbert":
        input_names = ["input_ids", "attention_mask"]
        dummy_inputs = (inputs["input_ids"], inputs["attention_mask"])
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        }
    else:
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"]))
        dummy_inputs = (inputs["input_ids"], inputs["attention_mask"], token_type_ids)
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        }

    # Export to ONNX FP32
    onnx_path = output_dir / "model.onnx"
    print("  Exporting to ONNX (FP32)...")
    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    fp32_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  FP32 model: {fp32_size:.1f} MB")

    # INT8 quantization
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantized_path = output_dir / "model_quantized.onnx"
    print("  Quantizing to INT8...")
    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QInt8,
    )
    int8_size = quantized_path.stat().st_size / (1024 * 1024)
    reduction = (1 - int8_size / fp32_size) * 100
    print(f"  INT8 model: {int8_size:.1f} MB ({reduction:.0f}% smaller)")

    # Copy supporting files
    print("  Copying supporting files...")
    for filename in COPY_FILES:
        src = model_dir / filename
        if src.exists():
            shutil.copy2(str(src), str(output_dir / filename))
            print(f"    Copied: {filename}")

    # Verify exported model
    print("  Verifying quantized model...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    ort_input_names = [i.name for i in session.get_inputs()]

    np_inputs = tokenizer(dummy_text, return_tensors="np", padding=True, truncation=True, max_length=128)
    feed = {"input_ids": np_inputs["input_ids"], "attention_mask": np_inputs["attention_mask"]}
    if "token_type_ids" in ort_input_names:
        feed["token_type_ids"] = np_inputs.get("token_type_ids", np.zeros_like(np_inputs["input_ids"]))

    outputs = session.run(None, feed)
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    best_idx = int(np.argmax(probs))
    best_label = id2label.get(str(best_idx), f"label_{best_idx}")
    confidence = float(probs[best_idx])

    print(f"  Test: '{dummy_text}' -> {best_label} ({confidence:.1%})")
    print(f"  [OK] {model_name} exported!")
    return True


def main():
    print("=" * 60)
    print("Hierarchical Models -> ONNX Export (Primary + Secondary)")
    print("=" * 60)

    if not HIERARCHY_DIR.exists():
        print(f"[ERROR] Hierarchy directory not found: {HIERARCHY_DIR}")
        sys.exit(1)

    # Clean output directory
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing output: {OUTPUT_DIR}")
        shutil.rmtree(str(OUTPUT_DIR))

    results = {}

    # 1. Export primary model
    primary_src = HIERARCHY_DIR / "primary" / "final"
    primary_out = OUTPUT_DIR / "primary"
    results["primary"] = export_model_to_onnx(primary_src, primary_out, "PRIMARY (8 categories)")

    # 2. Export each secondary model
    secondary_dir = HIERARCHY_DIR / "secondary"
    categories = sorted([
        d.name for d in secondary_dir.iterdir()
        if d.is_dir() and (d / "final" / "model.safetensors").exists()
    ])

    print(f"\nFound {len(categories)} secondary models: {categories}")

    for category in categories:
        src = secondary_dir / category / "final"
        out = OUTPUT_DIR / "secondary" / category
        results[f"secondary/{category}"] = export_model_to_onnx(
            src, out, f"SECONDARY: {category}"
        )

    # Copy metadata files
    metadata_src = HIERARCHY_DIR / "hierarchy_metadata.json"
    if metadata_src.exists():
        shutil.copy2(str(metadata_src), str(OUTPUT_DIR / "hierarchy_metadata.json"))
        print("\nCopied hierarchy_metadata.json")

    secondary_meta = HIERARCHY_DIR / "secondary" / "secondary_metadata.json"
    if secondary_meta.exists():
        shutil.copy2(str(secondary_meta), str(OUTPUT_DIR / "secondary_metadata.json"))
        print("Copied secondary_metadata.json")

    # Summary
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    success = sum(1 for v in results.values() if v)

    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {success}/{total} successful")

    if success < total:
        print(f"  {total - success} FAILED!")
        sys.exit(1)

    # Show total output size
    print(f"\nOutput in {OUTPUT_DIR}:")
    total_size = 0
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            rel = f.relative_to(OUTPUT_DIR)
            print(f"  {rel}: {size / 1024:.1f} KB")
    print(f"\nTotal output size: {total_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
