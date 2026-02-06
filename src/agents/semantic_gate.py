"""
Semantic Gate (Stage 1) - Pre-Classification Filtering

Filters out off-topic messages before they reach the BERT classifier using
per-category similarity thresholds and centroids.

Uses a local ONNX model (all-MiniLM-L6-v2) — no HuggingFace downloads.

Flow:
1. Compute message embedding via local ONNX model
2. Compare to each category centroid
3. Find best matching category
4. Check if similarity >= category-specific threshold
5. Block if below threshold, pass otherwise

Thresholds are loaded from training/results/semantic_gate_tuning.json
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class SemanticGate:
    """
    Semantic gate for filtering off-topic messages using per-category thresholds.

    Uses a local ONNX model for embeddings and cosine similarity to compare
    messages against category-specific centroids. No internet downloads needed.
    """

    # Mapping from MessageCategory enum values to tuning JSON keys
    CATEGORY_MAPPING = {
        "rag_query": "rag_queries",  # Note: tuning uses plural
        "professional": "professional",
        "psychological": "psychological",
        "learning": "learning",
        "social": "social",
        "emotional": "emotional",
        "aspirational": "aspirational",
        "chitchat": "chitchat",
        "off_topic": "off_topic",
    }

    def __init__(self, tuning_results_path: str = None, model_name: str = "all-MiniLM-L6-v2", training_data_dir: str = None, model_dir: str = None):
        """
        Initialize semantic gate with per-category thresholds and centroids.

        Args:
            tuning_results_path: Path to semantic_gate_tuning.json
            model_name: Model name (for logging only, model loaded from local ONNX)
            training_data_dir: Directory with training data files
            model_dir: Directory containing centroids.pkl
        """
        # Default paths (relative to project root)
        project_root = Path(__file__).parent.parent.parent
        if tuning_results_path is None:
            tuning_results_path = project_root / "training" / "results" / "semantic_gate_tuning.json"
        if training_data_dir is None:
            training_data_dir = project_root / "training" / "data" / "processed"
        if model_dir is None:
            from src.config import INTENT_CLASSIFIER_MODEL_PATH

            model_dir = project_root / INTENT_CLASSIFIER_MODEL_PATH
        model_dir = Path(model_dir)

        # Load tuning results
        self.tuning_results = self._load_tuning_results(tuning_results_path)
        self.category_thresholds = self.tuning_results["recommendation"]["thresholds"]

        # --- Load local ONNX model for embeddings ---
        onnx_model_dir = project_root / "training" / "models" / "onnx" / "model_int8"
        onnx_file = onnx_model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = onnx_model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"Semantic gate ONNX model not found in {onnx_model_dir}. Expected model_quantized.onnx or model.onnx")

        print(f"[SEMANTIC GATE] Loading ONNX model from {onnx_file}...")
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_model_dir))
        self.model_name = model_name
        print("[SEMANTIC GATE] ONNX model loaded (local, no download)")

        # Centroids: load from model_dir/centroids.pkl if available, else compute
        centroids_path = model_dir / "centroids.pkl"
        stored = self._load_stored_centroids(centroids_path)
        if stored is not None:
            self.category_centroids = stored
            print(f"[SEMANTIC GATE] Loaded centroids from {centroids_path}")
        else:
            print(f"[SEMANTIC GATE] Computing category centroids from {training_data_dir}...")
            self.category_centroids = self._compute_centroids(training_data_dir)
            self._save_centroids(centroids_path, self.category_centroids)
            print(f"[SEMANTIC GATE] Saved centroids to {centroids_path}")

        print("[SEMANTIC GATE] Initialization complete")
        print(f"  Loaded {len(self.category_thresholds)} category thresholds")
        print(f"  Centroids: {len(self.category_centroids)} categories")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using local ONNX model (mean pooling + L2 norm).

        Args:
            texts: List of strings to encode

        Returns:
            Array of shape (len(texts), 384) with normalized embeddings
        """
        all_embeddings = []
        for text in texts:
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

    def _load_tuning_results(self, path: str) -> dict:
        """Load tuning results from JSON file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Semantic gate tuning results not found at {path}. Run: python training/scripts/tune_semantic_gate.py")

        with open(path, "r") as f:
            results = json.load(f)

        print(f"[SEMANTIC GATE] Loaded tuning results from {path}")
        print(f"  Model: {results['model_name']}")
        print(f"  Mean domain acceptance: {results['global_metrics']['mean_domain_acceptance'] * 100:.2f}%")
        print(f"  Mean off-topic rejection: {results['global_metrics']['mean_offtopic_rejection'] * 100:.2f}%")

        return results

    def _load_stored_centroids(self, path: Path) -> Dict[str, np.ndarray] | None:
        """Load pre-computed centroids from centroids.pkl."""
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                centroids = pickle.load(f)
            if not centroids or not isinstance(centroids, dict):
                return None
            result = {}
            for cat, arr in centroids.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                result[cat] = arr
            return result
        except Exception as e:
            print(f"[SEMANTIC GATE] Could not load stored centroids: {e}")
            return None

    def _save_centroids(self, path: Path, centroids: Dict[str, np.ndarray]) -> None:
        """Save computed centroids to centroids.pkl for future use."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(centroids, f)
        except Exception as e:
            print(f"[SEMANTIC GATE] Could not save centroids: {e}")

    def _load_messages_from_file(self, file_path: str) -> List[str]:
        """Load messages from a text file (one per line)"""
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            messages = [line.strip() for line in f if line.strip()]
        return messages

    def _compute_centroids(self, data_dir: str) -> Dict[str, np.ndarray]:
        """Compute per-category centroids from training data using local ONNX model."""
        centroids = {}

        category_files = {
            "rag_queries": "rag_queries.txt",
            "professional": "professional.txt",
            "psychological": "psychological.txt",
            "learning": "learning.txt",
            "social": "social.txt",
            "emotional": "emotional.txt",
            "aspirational": "aspirational.txt",
            "chitchat": "chitchat.txt",
        }

        for category, filename in category_files.items():
            file_path = os.path.join(data_dir, filename)
            messages = self._load_messages_from_file(file_path)

            if not messages:
                print(f"  WARNING: No messages found for {category} at {file_path}")
                continue

            print(f"  Computing centroid for {category} ({len(messages)} messages)...")
            embeddings = self._encode(messages)
            centroid = np.mean(embeddings, axis=0, keepdims=True)
            centroids[category] = centroid

        return centroids

    def check_message(self, message: str, predicted_category: str, classifier_confidence: float = 0.0) -> Tuple[bool, float, str]:
        """
        Check if message should pass the semantic gate.

        Args:
            message: The user message
            predicted_category: Category predicted by the intent classifier
            classifier_confidence: Confidence score from the classifier (0-1).
                If high (>0.85), the threshold is halved to trust the classifier
                for short/specific messages that have low embedding similarity.

        Returns:
            Tuple of (should_pass, similarity, best_category)
        """
        message_embedding = self._encode([message])

        best_category = None
        best_similarity = -1

        for category, centroid in self.category_centroids.items():
            # Cosine similarity (both are normalized, so dot product = cosine sim)
            similarity = float(np.dot(message_embedding[0], centroid[0]))

            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        # Map predicted category to tuning key format
        predicted_category_key = self.CATEGORY_MAPPING.get(predicted_category, predicted_category)

        # Get threshold for the predicted category
        threshold_category = predicted_category_key if predicted_category_key in self.category_thresholds else best_category
        threshold = self.category_thresholds.get(threshold_category, 0.4)

        # If classifier is highly confident, reduce threshold (trust the classifier
        # for short/specific messages like product names with low embedding similarity)
        if classifier_confidence > 0.85:
            threshold *= 0.5

        should_pass = best_similarity >= threshold

        return should_pass, float(best_similarity), best_category

    def get_threshold(self, category: str, classifier_confidence: float = 0.0) -> float:
        """Get threshold for a specific category (adjusted for classifier confidence)."""
        category_key = self.CATEGORY_MAPPING.get(category, category)
        threshold = self.category_thresholds.get(category_key, 0.4)
        if classifier_confidence > 0.85:
            threshold *= 0.5
        return threshold

    def get_statistics(self) -> dict:
        """Get semantic gate statistics"""
        return {
            "model_name": self.model_name,
            "num_categories": len(self.category_centroids),
            "thresholds": self.category_thresholds,
            "global_metrics": self.tuning_results["global_metrics"],
        }


# =============================================================================
# Global Singleton (Lazy-loaded)
# =============================================================================

_semantic_gate_instance = None


def get_semantic_gate(force_reload: bool = False) -> SemanticGate:
    """Get or create the global semantic gate instance (singleton pattern)."""
    global _semantic_gate_instance

    if _semantic_gate_instance is None or force_reload:
        _semantic_gate_instance = SemanticGate()

    return _semantic_gate_instance
