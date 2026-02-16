"""
Export all hierarchical classifier models to ONNX format.

Converts 4 levels of models:
  - Routing (8 classes, softmax): professional, learning, social, psychological, personal, rag_query, chitchat, off_topic
  - Contexts (5 classes, softmax): professional, learning, social, psychological, personal
  - Entities per context (N classes, softmax): e.g., professional has 6 entities
  - Sub-entities per entity (M labels, sigmoid multi-label): e.g., professional_aspirations has 5 sub-entities

Each model is exported to ONNX FP32, then quantized to INT8.

Usage:
    python -m training.scripts.export_hierarchy_onnx
    python -m training.scripts.export_hierarchy_onnx --input-dir training/models/hierarchical --output-dir training/models/full_onnx

Input structure (from train_multilabel.py):
    training/models/hierarchical/
    ├── routing/final/                    (8-class softmax)
    ├── contexts/final/                   (5-class softmax)
    ├── professional/entities/final/      (N-class softmax)
    ├── professional/current_position/final/   (multi-label sigmoid)
    ├── professional/professional_aspirations/final/
    ├── learning/entities/final/
    ├── learning/current_skills/final/
    └── ...

Output structure:
    training/models/full_onnx/
    ├── hierarchy_metadata.json           (full hierarchy description)
    ├── routing/
    │   ├── model.onnx
    │   ├── model_quantized.onnx
    │   ├── config.json
    │   ├── label_mappings.json
    │   ├── tokenizer.json
    │   └── tokenizer_config.json
    ├── contexts/
    ├── professional/
    │   ├── entities/
    │   ├── current_position/
    │   ├── professional_aspirations/
    │   └── ...
    ├── learning/
    │   ├── entities/
    │   └── ...
    └── ...
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Files to copy alongside each ONNX model
COPY_FILES = [
    "config.json",
    "label_mappings.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

CONTEXTS = ["professional", "learning", "social", "psychological", "personal"]


def export_model_to_onnx(model_dir: Path, output_dir: Path, model_name: str) -> bool:
    """Export a single model to ONNX with INT8 quantization."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Source:  {model_dir}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")

    if not model_dir.exists():
        print(f"  [SKIP] Source directory not found: {model_dir}")
        return False

    # Check for model weights
    safetensors_file = model_dir / "model.safetensors"
    pytorch_file = model_dir / "pytorch_model.bin"
    if not safetensors_file.exists() and not pytorch_file.exists():
        print(f"  [SKIP] No model weights found in {model_dir}")
        return False

    # Load config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print(f"  [SKIP] No config.json found in {model_dir}")
        return False

    with open(config_file) as f:
        config = json.load(f)

    model_type = config.get("model_type", "unknown")
    architectures = config.get("architectures", [])
    id2label = config.get("id2label", {})
    num_labels = len(id2label)
    problem_type = config.get("problem_type", "single_label_classification")

    print(f"  Type: {model_type} ({architectures[0] if architectures else 'unknown'})")
    print(f"  Labels ({num_labels}): {list(id2label.values())}")
    print(f"  Problem: {problem_type}")

    # Load label_mappings.json for multi-label info
    label_mappings_file = model_dir / "label_mappings.json"
    label_config = {}
    if label_mappings_file.exists():
        with open(label_mappings_file) as f:
            label_config = json.load(f)

    is_multilabel = problem_type == "multi_label_classification" or label_config.get("problem_type") == "multi_label_classification"
    if is_multilabel:
        print(f"  Mode: MULTI-LABEL (sigmoid, threshold={label_config.get('threshold', 0.5)})")
    else:
        print(f"  Mode: SINGLE-LABEL (softmax)")

    # Load model and tokenizer
    print("  Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dummy input
    dummy_text = "This is a test message for ONNX export"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Determine input names based on model type
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
    for filename in COPY_FILES:
        src = model_dir / filename
        if src.exists():
            shutil.copy2(str(src), str(output_dir / filename))

    # Verify exported model
    print("  Verifying quantized model...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    ort_input_names = [i.name for i in session.get_inputs()]

    np_inputs = tokenizer(dummy_text, return_tensors="np", padding=True, truncation=True, max_length=128)
    feed = {
        "input_ids": np_inputs["input_ids"].astype(np.int64),
        "attention_mask": np_inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in ort_input_names:
        tid = np_inputs.get("token_type_ids", np.zeros_like(np_inputs["input_ids"]))
        feed["token_type_ids"] = tid.astype(np.int64)

    outputs = session.run(None, feed)
    logits = outputs[0][0]

    if is_multilabel:
        # Sigmoid for multi-label
        probs = 1 / (1 + np.exp(-logits))
        threshold = label_config.get("threshold", 0.5)
        active = [id2label.get(str(i), f"label_{i}") for i, p in enumerate(probs) if p >= threshold]
        print(f"  Test: '{dummy_text[:40]}...' -> {active} (threshold={threshold})")
    else:
        # Softmax for single-label
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        best_idx = int(np.argmax(probs))
        best_label = id2label.get(str(best_idx), f"label_{best_idx}")
        confidence = float(probs[best_idx])
        print(f"  Test: '{dummy_text[:40]}...' -> {best_label} ({confidence:.1%})")

    print(f"  [OK] {model_name}")
    return True


def build_hierarchy_metadata(input_dir: Path, results: dict) -> dict:
    """Build metadata JSON describing the full hierarchy."""
    metadata = {
        "version": "2.0",
        "levels": {
            "routing": {
                "path": "routing/",
                "type": "single_label",
                "description": "Routes messages to context or non-context types",
            },
            "contexts": {
                "path": "contexts/",
                "type": "single_label",
                "description": "Confirms which of 5 contexts the message belongs to",
            },
            "entities": {
                "type": "single_label",
                "description": "Classifies entity within a context",
                "models": {},
            },
            "sub_entities": {
                "type": "multi_label",
                "description": "Classifies sub-entities within an entity (multi-label, sigmoid)",
                "models": {},
            },
        },
        "export_results": {name: ok for name, ok in results.items()},
    }

    # Populate entity and sub-entity paths
    for ctx in CONTEXTS:
        entities_dir = input_dir / ctx / "entities" / "final"
        if entities_dir.exists():
            label_file = entities_dir / "label_mappings.json"
            if label_file.exists():
                with open(label_file) as f:
                    labels = json.load(f)
                metadata["levels"]["entities"]["models"][ctx] = {
                    "path": f"{ctx}/entities/",
                    "labels": list(labels.values()) if isinstance(labels, dict) else labels,
                }

        # Sub-entities
        ctx_dir = input_dir / ctx
        if ctx_dir.exists():
            for entity_dir in sorted(ctx_dir.iterdir()):
                if entity_dir.name == "entities" or not entity_dir.is_dir():
                    continue
                final_dir = entity_dir / "final"
                if final_dir.exists() and (final_dir / "label_mappings.json").exists():
                    with open(final_dir / "label_mappings.json") as f:
                        label_config = json.load(f)
                    labels = label_config.get("label_mappings", label_config)
                    if ctx not in metadata["levels"]["sub_entities"]["models"]:
                        metadata["levels"]["sub_entities"]["models"][ctx] = {}
                    metadata["levels"]["sub_entities"]["models"][ctx][entity_dir.name] = {
                        "path": f"{ctx}/{entity_dir.name}/",
                        "labels": list(labels.values()) if isinstance(labels, dict) else labels,
                        "threshold": label_config.get("threshold", 0.5),
                    }

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export hierarchical models to ONNX")
    parser.add_argument("--input-dir", type=str, default="training/models/hierarchical",
                        help="Directory with trained models from train_multilabel.py")
    parser.add_argument("--output-dir", type=str, default="training/models/full_onnx",
                        help="Output directory for ONNX models")
    parser.add_argument("--keep-fp32", action="store_true",
                        help="Keep FP32 models (default: only keep quantized)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("HIERARCHICAL MODELS -> ONNX EXPORT")
    print("=" * 60)
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    if not input_dir.exists():
        print(f"\n[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    # Clean output directory
    if output_dir.exists():
        print(f"\n  Removing existing output: {output_dir}")
        shutil.rmtree(str(output_dir))

    results = {}

    # =========================================================================
    # 1. ROUTING (8 classes, softmax)
    # =========================================================================
    routing_src = input_dir / "routing" / "final"
    routing_out = output_dir / "routing"
    results["routing"] = export_model_to_onnx(routing_src, routing_out, "ROUTING (8 classes)")

    # =========================================================================
    # 2. CONTEXTS (5 classes, softmax)
    # =========================================================================
    contexts_src = input_dir / "contexts" / "final"
    contexts_out = output_dir / "contexts"
    results["contexts"] = export_model_to_onnx(contexts_src, contexts_out, "CONTEXTS (5 classes)")

    # =========================================================================
    # 3. ENTITIES per context (softmax)
    # =========================================================================
    for ctx in CONTEXTS:
        entity_src = input_dir / ctx / "entities" / "final"
        entity_out = output_dir / ctx / "entities"
        results[f"{ctx}/entities"] = export_model_to_onnx(
            entity_src, entity_out, f"{ctx.upper()} ENTITIES"
        )

    # =========================================================================
    # 4. SUB-ENTITIES per entity (sigmoid multi-label)
    # =========================================================================
    for ctx in CONTEXTS:
        ctx_dir = input_dir / ctx
        if not ctx_dir.exists():
            continue
        for entity_dir in sorted(ctx_dir.iterdir()):
            if entity_dir.name == "entities" or not entity_dir.is_dir():
                continue
            final_dir = entity_dir / "final"
            if not final_dir.exists():
                continue
            sub_out = output_dir / ctx / entity_dir.name
            results[f"{ctx}/{entity_dir.name}"] = export_model_to_onnx(
                final_dir, sub_out, f"{ctx.upper()} > {entity_dir.name.upper()} SUB-ENTITIES"
            )

    # =========================================================================
    # Build hierarchy metadata
    # =========================================================================
    metadata = build_hierarchy_metadata(input_dir, results)
    metadata_path = output_dir / "hierarchy_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved hierarchy_metadata.json")

    # Optionally remove FP32 models
    if not args.keep_fp32:
        print("\nRemoving FP32 models (keeping only quantized)...")
        for fp32 in output_dir.rglob("model.onnx"):
            quantized = fp32.parent / "model_quantized.onnx"
            if quantized.exists():
                fp32.unlink()
                print(f"  Removed: {fp32.relative_to(output_dir)}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    success = sum(1 for v in results.values() if v)
    skipped = sum(1 for v in results.values() if not v)

    for name, ok in sorted(results.items()):
        status = "OK" if ok else "SKIP"
        print(f"  [{status}] {name}")

    print(f"\n  Exported: {success}/{total}")
    if skipped:
        print(f"  Skipped: {skipped} (no trained model found)")

    # Total output size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\n  Total output size: {total_size / (1024*1024):.1f} MB")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
