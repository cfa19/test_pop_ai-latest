"""
Download all-MiniLM-L6-v2 ONNX model from HuggingFace for semantic gate.

This script:
1. Downloads the official pre-quantized ONNX model from sentence-transformers/all-MiniLM-L6-v2
2. Downloads the tokenizer files
3. Verifies the model produces valid embeddings

Output: training/models/onnx/semantic_gate/
  - model_quantized.onnx (pre-quantized INT8)
  - tokenizer.json, tokenizer_config.json, vocab.txt, etc.

Usage:
    python training/scripts/export_minilm_onnx.py

Requirements: pip install onnxruntime transformers huggingface_hub
"""

import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def export_minilm_onnx(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "training/models/onnx/semantic_gate",
):
    """Download official pre-quantized ONNX model and tokenizer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Download pre-quantized ONNX model
    print(f"[Export] Downloading pre-quantized ONNX model from {model_name}...")
    quantized_path = output_path / "model_quantized.onnx"
    downloaded = hf_hub_download(
        repo_id=model_name,
        filename="onnx/model_quint8_avx2.onnx",
        local_dir=str(output_path / "_tmp"),
    )
    shutil.copy2(downloaded, str(quantized_path))
    shutil.rmtree(str(output_path / "_tmp"), ignore_errors=True)
    print(f"[Export] Downloaded: {quantized_path} ({quantized_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 2: Save tokenizer
    print("[Export] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(output_path))

    # Step 3: Verify the model
    print("[Export] Verifying ONNX model...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    print(f"  Model inputs: {input_names}")

    test_inputs = tokenizer("Hello world", return_tensors="np", padding=True, truncation=True, max_length=128)
    feed = {
        "input_ids": test_inputs["input_ids"].astype(np.int64),
        "attention_mask": test_inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in input_names:
        tid = test_inputs.get("token_type_ids", np.zeros_like(test_inputs["input_ids"]))
        feed["token_type_ids"] = tid.astype(np.int64)

    outputs = session.run(None, feed)
    last_hidden_state = outputs[0]

    # Mean pool + normalize (same as semantic_gate_onnx.py)
    mask = test_inputs["attention_mask"]
    mask_exp = np.expand_dims(mask, -1).astype(np.float32)
    mask_exp = np.broadcast_to(mask_exp, last_hidden_state.shape)
    pooled = np.sum(last_hidden_state * mask_exp, axis=1) / np.clip(np.sum(mask_exp, axis=1), 1e-9, None)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    embedding = pooled / np.clip(norm, 1e-9, None)

    print(f"[Export] Verification passed!")
    print(f"  Output shape: {last_hidden_state.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")

    # Also download full precision model for fallback
    print(f"\n[Export] Also downloading full precision model...")
    full_path = output_path / "model.onnx"
    downloaded_full = hf_hub_download(
        repo_id=model_name,
        filename="onnx/model.onnx",
        local_dir=str(output_path / "_tmp"),
    )
    shutil.copy2(downloaded_full, str(full_path))
    shutil.rmtree(str(output_path / "_tmp"), ignore_errors=True)
    print(f"  Full model: {full_path} ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\n[Export] Done! Files saved to {output_path}/")
    for f in sorted(output_path.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name} ({size / 1024 / 1024:.1f} MB)")
        elif size > 1024:
            print(f"  {f.name} ({size / 1024:.1f} KB)")
        else:
            print(f"  {f.name} ({size} B)")


if __name__ == "__main__":
    export_minilm_onnx()
