"""
Re-tune Semantic Gate using ONNX model embeddings.

The original centroids were computed with PyTorch SentenceTransformer (full precision),
but production uses the ONNX quantized model (model_quint8_avx2.onnx). This causes
embedding drift and low similarity scores.

This script re-computes centroids and thresholds using the same ONNX model
that runs in production, ensuring consistency.

Usage:
    python training/scripts/retune_semantic_gate_onnx.py

Outputs:
    training/models/onnx/semantic_gate/primary_centroids.pkl
    training/models/onnx/semantic_gate/secondary_centroids.pkl
    training/results/semantic_gate_hierarchical_tuning.json
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# ONNX Embedding Model
# ============================================================================

class OnnxEmbedder:
    """Compute embeddings using the same ONNX model as production."""

    def __init__(self, model_dir: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = Path(model_dir)
        onnx_file = model_path / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_path / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"No ONNX model in {model_path}")

        print(f"  Loading ONNX model: {onnx_file}")
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print(f"  Model loaded. Inputs: {self.input_names}")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode list of texts to embeddings. Returns (N, dim) array."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="np", padding=True, truncation=True, max_length=128
            )
            feed = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = inputs.get(
                    "token_type_ids", np.zeros_like(inputs["input_ids"])
                )

            outputs = self.session.run(None, feed)
            last_hidden = outputs[0]  # (batch, seq_len, hidden_dim)

            # Mean pooling
            mask = inputs["attention_mask"]
            mask_expanded = np.expand_dims(mask, -1).astype(np.float32)
            mask_expanded = np.broadcast_to(mask_expanded, last_hidden.shape)
            summed = np.sum(last_hidden * mask_expanded, axis=1)
            counts = np.clip(np.sum(mask_expanded, axis=1), 1e-9, None)
            pooled = summed / counts

            # L2 normalize
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norms, 1e-9, None)
            all_embeddings.append(normalized)

            if (i + batch_size) % 500 < batch_size:
                print(f"    Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

        return np.vstack(all_embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text. Returns (1, dim)."""
        return self.encode([text])


# ============================================================================
# ONNX Secondary Classifier
# ============================================================================

class OnnxSecondaryClassifier:
    """Use ONNX secondary classifiers to predict subcategories."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self._cache = {}

    def classify(self, text: str, category: str) -> Tuple[str, float]:
        """Classify a single message into a subcategory."""
        if category not in self._cache:
            self._load_classifier(category)
        if category not in self._cache:
            return None, 0.0

        session, tokenizer, id2label, input_names = self._cache[category]
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
        feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

        outputs = session.run(None, feed)
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        best_idx = int(np.argmax(probs))
        return id2label.get(str(best_idx), f"label_{best_idx}"), float(probs[best_idx])

    def _load_classifier(self, category: str):
        """Load an ONNX secondary classifier for a category."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        cat_path = self.base_path / "secondary" / category
        onnx_file = cat_path / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = cat_path / "model.onnx"
        if not onnx_file.exists():
            print(f"  [WARN] No secondary classifier for {category}")
            return

        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(str(cat_path))
        input_names = [i.name for i in session.get_inputs()]

        config_file = cat_path / "config.json"
        with open(config_file) as f:
            config = json.load(f)
        id2label = config.get("id2label", {})

        self._cache[category] = (session, tokenizer, id2label, input_names)
        print(f"  Loaded secondary classifier: {category} ({len(id2label)} subcategories)")

    def get_subcategories(self, category: str) -> List[str]:
        """Get list of subcategories for a category."""
        if category not in self._cache:
            self._load_classifier(category)
        if category not in self._cache:
            return []
        _, _, id2label, _ = self._cache[category]
        return list(id2label.values())


# ============================================================================
# Data Loading
# ============================================================================

def load_messages(file_path: str) -> List[str]:
    """Load messages from text file (one per line)."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_all_domain_data(data_dir: str) -> Dict[str, List[str]]:
    """Load all domain category data files."""
    categories = [
        "rag_queries", "professional", "psychological", "learning",
        "social", "emotional", "aspirational", "chitchat"
    ]
    data = {}
    for cat in categories:
        msgs = load_messages(os.path.join(data_dir, f"{cat}.txt"))
        if msgs:
            # Normalize category name (rag_queries -> rag_query)
            key = "rag_query" if cat == "rag_queries" else cat
            data[key] = msgs
            print(f"  {key}: {len(msgs)} messages")
    return data


# ============================================================================
# Threshold Tuning
# ============================================================================

def evaluate_threshold(domain_sims, offtopic_sims, threshold):
    """Evaluate a threshold: returns metrics dict."""
    tp = int(np.sum(domain_sims >= threshold))
    fn = int(np.sum(domain_sims < threshold))
    tn = int(np.sum(offtopic_sims < threshold))
    fp = int(np.sum(offtopic_sims >= threshold))

    total_d = len(domain_sims)
    total_o = len(offtopic_sims)

    acc_rate = tp / total_d if total_d > 0 else 0
    rej_rate = tn / total_o if total_o > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = acc_rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold,
        "true_positives": tp, "false_negatives": fn,
        "true_negatives": tn, "false_positives": fp,
        "domain_acceptance_rate": acc_rate,
        "offtopic_rejection_rate": rej_rate,
        "precision": precision, "recall": recall, "f1_score": f1,
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        "total_domain": total_d, "total_offtopic": total_o,
    }


def find_optimal_threshold(domain_sims, offtopic_sims, min_acceptance=0.95):
    """Find best threshold with >=min_acceptance domain acceptance."""
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_t, best_m, best_score = None, None, -1

    for t in thresholds:
        m = evaluate_threshold(domain_sims, offtopic_sims, t)
        if m["domain_acceptance_rate"] >= min_acceptance:
            score = m["offtopic_rejection_rate"] * 0.7 + m["f1_score"] * 0.3
            if score > best_score:
                best_score = score
                best_t = round(t, 2)
                best_m = m

    if best_t is None:
        # Fallback: best F1
        best_f1 = -1
        for t in thresholds:
            m = evaluate_threshold(domain_sims, offtopic_sims, t)
            if m["f1_score"] > best_f1:
                best_f1 = m["f1_score"]
                best_t = round(t, 2)
                best_m = m

    return best_t, best_m


def cosine_similarities(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between embeddings and a centroid."""
    # Both should be L2-normalized already
    return (embeddings @ centroid.T).flatten()


# ============================================================================
# Main
# ============================================================================

def main():
    DATA_DIR = str(PROJECT_ROOT / "training" / "data" / "processed")
    ONNX_MODEL_DIR = str(PROJECT_ROOT / "training" / "models" / "onnx" / "semantic_gate")
    ONNX_HIERARCHY_DIR = str(PROJECT_ROOT / "training" / "models" / "full_onnx")
    OUTPUT_CENTROIDS_DIR = str(PROJECT_ROOT / "training" / "models" / "onnx" / "semantic_gate")
    OUTPUT_TUNING = str(PROJECT_ROOT / "training" / "results" / "semantic_gate_hierarchical_tuning.json")

    SKIP_SECONDARY = {"rag_query", "chitchat", "off_topic"}
    MIN_ACCEPTANCE = 0.95

    print("=" * 80)
    print("RE-TUNE SEMANTIC GATE USING ONNX MODEL")
    print("=" * 80)

    # 1. Load data
    print("\n[1/7] Loading training data...")
    domain_data = load_all_domain_data(DATA_DIR)
    offtopic_messages = load_messages(os.path.join(DATA_DIR, "off_topic.txt"))
    print(f"  Off-topic: {len(offtopic_messages)} messages")

    # 2. Load ONNX embedding model (same as production)
    print(f"\n[2/7] Loading ONNX embedding model from {ONNX_MODEL_DIR}...")
    embedder = OnnxEmbedder(ONNX_MODEL_DIR)

    # Quick sanity check
    test_emb = embedder.encode_single("I feel depressed")
    print(f"  Embedding dim: {test_emb.shape[1]}")
    print(f"  Norm: {np.linalg.norm(test_emb):.4f}")

    # 3. Load secondary classifiers (for subcategory grouping)
    print(f"\n[3/7] Loading ONNX secondary classifiers from {ONNX_HIERARCHY_DIR}...")
    secondary_clf = OnnxSecondaryClassifier(ONNX_HIERARCHY_DIR)

    # 4. Compute primary centroids
    print("\n[4/7] Computing primary centroids with ONNX model...")
    primary_centroids = {}
    primary_embeddings = {}  # Keep for threshold tuning

    for category, messages in domain_data.items():
        print(f"\n  [{category}] Encoding {len(messages)} messages...")
        embs = embedder.encode(messages)
        centroid = np.mean(embs, axis=0, keepdims=True)
        # L2 normalize centroid
        centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-9, None)
        primary_centroids[category] = centroid
        primary_embeddings[category] = embs
        print(f"  [{category}] Centroid norm: {np.linalg.norm(centroid):.4f}")

    # 5. Compute secondary centroids
    print("\n[5/7] Computing secondary centroids...")
    secondary_centroids = {}

    for category, messages in domain_data.items():
        if category in SKIP_SECONDARY:
            continue

        subcats = secondary_clf.get_subcategories(category)
        if not subcats:
            print(f"  [{category}] No secondary classifier found, skipping")
            continue

        print(f"\n  [{category}] Classifying {len(messages)} messages into subcategories...")
        grouped = defaultdict(list)
        grouped_indices = defaultdict(list)

        for idx, msg in enumerate(messages):
            subcat, conf = secondary_clf.classify(msg, category)
            if subcat:
                grouped[subcat].append(msg)
                grouped_indices[subcat].append(idx)

        # Compute centroid per subcategory using pre-computed embeddings
        cat_centroids = {}
        for subcat, indices in grouped_indices.items():
            subcat_embs = primary_embeddings[category][indices]
            centroid = np.mean(subcat_embs, axis=0, keepdims=True)
            centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-9, None)
            cat_centroids[subcat] = centroid
            print(f"    {subcat}: {len(indices)} messages, centroid norm: {np.linalg.norm(centroid):.4f}")

        secondary_centroids[category] = cat_centroids

    # 6. Tune thresholds
    print("\n[6/7] Tuning thresholds...")

    # Classify off-topic messages using primary ONNX classifier
    print(f"\n  Encoding {len(offtopic_messages)} off-topic messages...")
    offtopic_embs = embedder.encode(offtopic_messages)

    # Group off-topic by closest primary centroid
    print("  Grouping off-topic messages by closest primary centroid...")
    offtopic_grouped = defaultdict(list)
    offtopic_grouped_embs = defaultdict(list)

    for idx in range(len(offtopic_messages)):
        best_cat = None
        best_sim = -1
        emb = offtopic_embs[idx:idx+1]
        for cat, centroid in primary_centroids.items():
            sim = float(cosine_similarities(emb, centroid)[0])
            if sim > best_sim:
                best_sim = sim
                best_cat = cat
        offtopic_grouped[best_cat].append(offtopic_messages[idx])
        offtopic_grouped_embs[best_cat].append(idx)

    for cat, msgs in sorted(offtopic_grouped.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"    {cat}: {len(msgs)} off-topic messages")

    # Tune primary thresholds
    print("\n  Tuning primary thresholds...")
    primary_results = {}
    primary_thresholds = {}

    for category in domain_data.keys():
        domain_embs = primary_embeddings[category]
        cat_centroid = primary_centroids[category]

        domain_sims = cosine_similarities(domain_embs, cat_centroid)
        ot_indices = offtopic_grouped_embs.get(category, [])
        if ot_indices:
            ot_embs = offtopic_embs[ot_indices]
            ot_sims = cosine_similarities(ot_embs, cat_centroid)
        else:
            # Use all off-topic
            ot_sims = cosine_similarities(offtopic_embs, cat_centroid)

        threshold, metrics = find_optimal_threshold(domain_sims, ot_sims, MIN_ACCEPTANCE)
        primary_thresholds[category] = threshold

        print(f"    {category}: threshold={threshold:.2f}, "
              f"accept={metrics['domain_acceptance_rate']*100:.1f}%, "
              f"reject={metrics['offtopic_rejection_rate']*100:.1f}%, "
              f"mean_sim={np.mean(domain_sims):.4f}")

        primary_results[category] = {
            "threshold": threshold,
            "metrics": metrics,
            "domain_similarity_stats": {
                "mean": float(np.mean(domain_sims)),
                "std": float(np.std(domain_sims)),
            }
        }

    # Tune secondary thresholds
    print("\n  Tuning secondary thresholds...")
    secondary_thresholds = {}
    secondary_results = {}

    for category, cat_centroids in secondary_centroids.items():
        secondary_thresholds[category] = {}
        secondary_results[category] = {}

        # Get off-topic embeddings for this category
        ot_indices = offtopic_grouped_embs.get(category, [])
        if ot_indices:
            ot_embs = offtopic_embs[ot_indices]
        else:
            ot_embs = offtopic_embs

        for subcat, centroid in cat_centroids.items():
            # Domain: messages classified as this subcategory
            subcat_embs = []
            for idx, msg in enumerate(domain_data[category]):
                pred_sub, _ = secondary_clf.classify(msg, category)
                if pred_sub == subcat:
                    subcat_embs.append(primary_embeddings[category][idx])

            if len(subcat_embs) < 10:
                print(f"    {category}/{subcat}: too few messages ({len(subcat_embs)}), using primary threshold")
                secondary_thresholds[category][subcat] = primary_thresholds[category]
                continue

            subcat_embs = np.vstack(subcat_embs)
            domain_sims = cosine_similarities(subcat_embs, centroid)
            ot_sims = cosine_similarities(ot_embs, centroid)

            threshold, metrics = find_optimal_threshold(domain_sims, ot_sims, MIN_ACCEPTANCE)
            secondary_thresholds[category][subcat] = threshold

            print(f"    {category}/{subcat}: threshold={threshold:.2f}, "
                  f"accept={metrics['domain_acceptance_rate']*100:.1f}%, "
                  f"reject={metrics['offtopic_rejection_rate']*100:.1f}%")

            secondary_results[category][subcat] = {
                "threshold": threshold,
                "metrics": metrics,
                "domain_similarity_stats": {
                    "mean": float(np.mean(domain_sims)),
                    "std": float(np.std(domain_sims)),
                }
            }

    # 7. Save everything
    print("\n[7/7] Saving results...")

    # Save centroids
    os.makedirs(OUTPUT_CENTROIDS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_CENTROIDS_DIR, "primary_centroids.pkl"), "wb") as f:
        pickle.dump(primary_centroids, f)
    print(f"  Saved primary centroids: {len(primary_centroids)} categories")

    with open(os.path.join(OUTPUT_CENTROIDS_DIR, "secondary_centroids.pkl"), "wb") as f:
        pickle.dump(secondary_centroids, f)
    total_sec = sum(len(v) for v in secondary_centroids.values())
    print(f"  Saved secondary centroids: {total_sec} subcategories")

    # Save tuning results (same format as original)
    global_metrics = {
        "primary": {
            "mean_threshold": float(np.mean(list(primary_thresholds.values()))),
            "mean_domain_acceptance": float(np.mean([r["metrics"]["domain_acceptance_rate"] for r in primary_results.values()])),
            "mean_offtopic_rejection": float(np.mean([r["metrics"]["offtopic_rejection_rate"] for r in primary_results.values()])),
            "mean_f1_score": float(np.mean([r["metrics"]["f1_score"] for r in primary_results.values()])),
        }
    }

    tuning = {
        "model_name": "all-MiniLM-L6-v2 (ONNX quantized)",
        "model_path": ONNX_MODEL_DIR,
        "approach": "hierarchical_thresholds",
        "primary_categories": len(primary_centroids),
        "secondary_categories": total_sec,
        "global_metrics": global_metrics,
        "primary_thresholds": primary_thresholds,
        "secondary_thresholds": secondary_thresholds,
        "detailed_results": {
            "primary": primary_results,
            "secondary": secondary_results,
        },
        "offtopic_distribution": {
            "primary": {cat: len(msgs) for cat, msgs in offtopic_grouped.items()},
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_TUNING), exist_ok=True)
    with open(OUTPUT_TUNING, "w") as f:
        json.dump(tuning, f, indent=2)
    print(f"  Saved tuning results: {OUTPUT_TUNING}")

    # Quick verification
    print("\n" + "=" * 80)
    print("VERIFICATION: Testing 'I feel depressed' against new centroids")
    print("=" * 80)
    test_emb = embedder.encode_single("I feel depressed")
    for cat, centroid in primary_centroids.items():
        sim = float(cosine_similarities(test_emb, centroid)[0])
        threshold = primary_thresholds[cat]
        status = "PASS" if sim >= threshold else "FAIL"
        print(f"  {cat:20s}: sim={sim:.4f} vs threshold={threshold:.2f} [{status}]")

    print("\n  Primary thresholds summary:")
    for cat, t in sorted(primary_thresholds.items()):
        print(f"    {cat}: {t}")

    print(f"\n  Global acceptance: {global_metrics['primary']['mean_domain_acceptance']*100:.1f}%")
    print(f"  Global rejection: {global_metrics['primary']['mean_offtopic_rejection']*100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()
