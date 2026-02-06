"""
Re-tune Semantic Gate thresholds for local ONNX model.

The original thresholds were calibrated with SentenceTransformer (HuggingFace).
This script re-calibrates them using the local ONNX model (model_int8) that
the semantic gate actually uses in production.

Usage:
    python training/scripts/retune_semantic_gate_onnx.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_messages(file_path: str, max_messages: int = 0) -> List[str]:
    """Load messages from a text file (one per line)."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        messages = [line.strip() for line in f if line.strip()]
    if max_messages > 0:
        messages = messages[:max_messages]
    return messages


class ONNXEncoder:
    """Encode text using local ONNX model (mean pooling + L2 norm)."""

    def __init__(self, model_dir: Path):
        onnx_file = model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX model not found in {model_dir}")

        print(f"Loading ONNX model from {onnx_file}...")
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        print("ONNX model loaded.")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        all_embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 200 == 0:
                print(f"  Encoded {i + 1}/{len(texts)} messages...")

            inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
            feed = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

            outputs = self.session.run(None, feed)
            hidden = outputs[0]  # (1, seq_len, 384)

            # Mean pooling
            mask = inputs["attention_mask"][..., np.newaxis]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)

            # L2 normalize
            embedding = pooled[0]
            embedding = embedding / np.maximum(np.linalg.norm(embedding), 1e-9)
            all_embeddings.append(embedding)

        return np.array(all_embeddings)


def evaluate_threshold(domain_sims: np.ndarray, offtopic_sims: np.ndarray, threshold: float) -> Dict:
    """Evaluate performance at a specific threshold."""
    tp = int(np.sum(domain_sims >= threshold))
    fn = int(np.sum(domain_sims < threshold))
    tn = int(np.sum(offtopic_sims < threshold))
    fp = int(np.sum(offtopic_sims >= threshold))

    total_d = len(domain_sims)
    total_o = len(offtopic_sims)

    dar = tp / total_d if total_d > 0 else 0
    orr = tn / total_o if total_o > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "domain_acceptance_rate": dar,
        "offtopic_rejection_rate": orr,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy": acc,
        "total_domain": total_d,
        "total_offtopic": total_o,
    }


def find_optimal_threshold(domain_sims: np.ndarray, offtopic_sims: np.ndarray, min_domain_acceptance: float = 0.95) -> Tuple[float, Dict]:
    """Find optimal threshold: accepts >=95% domain, maximizes off-topic rejection."""
    thresholds = np.linspace(0.0, 1.0, 201)  # 0.005 step

    best_threshold = 0.0
    best_metrics = None
    best_score = -1

    for t in thresholds:
        m = evaluate_threshold(domain_sims, offtopic_sims, t)
        if m["domain_acceptance_rate"] >= min_domain_acceptance:
            score = m["offtopic_rejection_rate"] * 0.7 + m["f1_score"] * 0.3
            if score > best_score:
                best_score = score
                best_threshold = t
                best_metrics = m

    if best_metrics is None:
        # Fallback: best F1
        best_f1 = -1
        for t in thresholds:
            m = evaluate_threshold(domain_sims, offtopic_sims, t)
            if m["f1_score"] > best_f1:
                best_f1 = m["f1_score"]
                best_threshold = t
                best_metrics = m

    return best_threshold, best_metrics


def main():
    data_dir = PROJECT_ROOT / "training" / "data" / "processed"
    onnx_model_dir = PROJECT_ROOT / "training" / "models" / "onnx" / "model_int8"
    output_path = PROJECT_ROOT / "training" / "results" / "semantic_gate_tuning.json"

    categories = ["rag_queries", "professional", "psychological", "learning", "social", "emotional", "aspirational", "chitchat"]

    print("=" * 70)
    print("RE-TUNING SEMANTIC GATE THRESHOLDS FOR ONNX MODEL")
    print("=" * 70)

    # 1. Load encoder
    encoder = ONNXEncoder(onnx_model_dir)

    # 2. Load domain data
    print("\nLoading domain data...")
    domain_messages = {}
    for cat in categories:
        msgs = load_messages(str(data_dir / f"{cat}.txt"))
        if msgs:
            domain_messages[cat] = msgs
            print(f"  {cat}: {len(msgs)} messages")

    # 3. Load off-topic data
    offtopic_msgs = load_messages(str(data_dir / "off_topic.txt"))
    print(f"  off_topic: {len(offtopic_msgs)} messages")

    # 4. Compute centroids per category
    print("\nComputing per-category centroids...")
    centroids = {}
    for cat, msgs in domain_messages.items():
        print(f"  Encoding {cat} ({len(msgs)} messages)...")
        embs = encoder.encode(msgs)
        centroids[cat] = np.mean(embs, axis=0, keepdims=True)
        print("    Centroid computed.")

    # 5. Compute off-topic embeddings
    print(f"\nEncoding off-topic messages ({len(offtopic_msgs)})...")
    offtopic_embs = encoder.encode(offtopic_msgs)
    print("  Done.")

    # 6. Tune per-category thresholds
    print("\n" + "=" * 70)
    print("TUNING PER-CATEGORY THRESHOLDS")
    print("=" * 70)

    category_results = {}

    for cat in categories:
        if cat not in domain_messages:
            continue

        print(f"\n[{cat.upper()}]")

        # Domain embeddings
        domain_embs = encoder.encode(domain_messages[cat])

        # Similarity to category centroid
        centroid = centroids[cat]
        domain_sims = np.dot(domain_embs, centroid[0])  # cosine sim (both L2-normed)
        offtopic_sims = np.dot(offtopic_embs, centroid[0])

        print(f"  Domain similarity  - Mean: {np.mean(domain_sims):.4f}, Std: {np.std(domain_sims):.4f}")
        print(f"  Off-topic similarity - Mean: {np.mean(offtopic_sims):.4f}, Std: {np.std(offtopic_sims):.4f}")

        # Find optimal threshold
        threshold, metrics = find_optimal_threshold(domain_sims, offtopic_sims)

        print(f"  Optimal threshold: {threshold:.4f}")
        print(f"  Domain acceptance: {metrics['domain_acceptance_rate'] * 100:.2f}%")
        print(f"  Off-topic rejection: {metrics['offtopic_rejection_rate'] * 100:.2f}%")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

        category_results[cat] = {
            "threshold": threshold,
            "metrics": metrics,
            "domain_similarity_stats": {
                "mean": float(np.mean(domain_sims)),
                "std": float(np.std(domain_sims)),
                "min": float(np.min(domain_sims)),
                "max": float(np.max(domain_sims)),
                "percentile_5": float(np.percentile(domain_sims, 5)),
            },
            "offtopic_similarity_stats": {
                "mean": float(np.mean(offtopic_sims)),
                "std": float(np.std(offtopic_sims)),
                "min": float(np.min(offtopic_sims)),
                "max": float(np.max(offtopic_sims)),
                "percentile_95": float(np.percentile(offtopic_sims, 95)),
            },
            "offtopic_count": len(offtopic_msgs),
        }

    # 7. Save results
    global_metrics = {
        "mean_threshold": float(np.mean([r["threshold"] for r in category_results.values()])),
        "mean_domain_acceptance": float(np.mean([r["metrics"]["domain_acceptance_rate"] for r in category_results.values()])),
        "mean_offtopic_rejection": float(np.mean([r["metrics"]["offtopic_rejection_rate"] for r in category_results.values()])),
        "mean_f1_score": float(np.mean([r["metrics"]["f1_score"] for r in category_results.values()])),
    }

    results = {
        "model_name": "all-MiniLM-L6-v2-onnx-int8",
        "use_per_category_thresholds": True,
        "global_metrics": global_metrics,
        "per_category_results": category_results,
        "offtopic_distribution": {cat: len(offtopic_msgs) for cat in categories if cat in domain_messages},
        "recommendation": {
            "approach": "per-category thresholds",
            "thresholds": {cat: r["threshold"] for cat, r in category_results.items()},
            "usage": "Use category-specific thresholds in semantic gate configuration",
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # 8. Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Category':<20} {'Threshold':<12} {'Domain Acc':<12} {'OT Reject':<12} {'F1':<10}")
    print("-" * 66)
    for cat, r in sorted(category_results.items()):
        m = r["metrics"]
        print(
            f"{cat:<20} {r['threshold']:<12.4f} {m['domain_acceptance_rate'] * 100:<11.2f}% "
            f"{m['offtopic_rejection_rate'] * 100:<11.2f}% {m['f1_score']:<10.4f}"
        )

    print("-" * 66)
    print(
        f"{'AVERAGE':<20} {global_metrics['mean_threshold']:<12.4f} "
        f"{global_metrics['mean_domain_acceptance'] * 100:<11.2f}% "
        f"{global_metrics['mean_offtopic_rejection'] * 100:<11.2f}% "
        f"{global_metrics['mean_f1_score']:<10.4f}"
    )

    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
