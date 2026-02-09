"""
ONNX-based Hierarchical Semantic Gate - Two-Level Filtering (No PyTorch Required)

Drop-in replacement for semantic_gate.py that uses ONNX Runtime instead of
sentence-transformers + PyTorch. Uses ~25-40MB RAM instead of ~1GB+.

Filters off-topic messages using hierarchical thresholds:
- Primary level: Category thresholds (rag_query, professional, etc.)
- Secondary level: Subcategory thresholds (skills, experience, etc.)

Data Sources:
- ONNX Model: {model_path}/model_quantized.onnx (exported from all-MiniLM-L6-v2)
- Tokenizer: {model_path}/tokenizer.json etc.
- Thresholds: training/results/semantic_gate_hierarchical_tuning.json
- Centroids: {centroids_dir}/primary_centroids.pkl, secondary_centroids.pkl
"""

import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class SemanticGateONNX:
    """
    ONNX-based hierarchical semantic gate for filtering off-topic messages.

    Uses ONNX Runtime for MiniLM embeddings and numpy for cosine similarity.
    No PyTorch or sentence-transformers required.
    """

    CATEGORY_MAPPING = {
        "rag_query": "rag_query",
        "professional": "professional",
        "psychological": "psychological",
        "learning": "learning",
        "social": "social",
        "emotional": "emotional",
        "aspirational": "aspirational",
        "chitchat": "chitchat",
        "off_topic": "off_topic",
    }

    SKIP_SECONDARY = {"rag_query", "chitchat", "off_topic"}

    def __init__(self, model_path: str = None, tuning_results_path: str = None, centroids_dir: str = None):
        """
        Initialize ONNX semantic gate.

        Args:
            model_path: Path to directory with model_quantized.onnx + tokenizer files
            tuning_results_path: Path to semantic_gate_hierarchical_tuning.json
            centroids_dir: Directory containing primary_centroids.pkl and secondary_centroids.pkl
        """
        project_root = Path(__file__).parent.parent.parent

        # Resolve model path
        if model_path is None:
            from src.config import SEMANTIC_GATE_ONNX_MODEL_PATH
            model_path = SEMANTIC_GATE_ONNX_MODEL_PATH
        model_dir = Path(model_path)

        # Resolve tuning results path
        if tuning_results_path is None:
            from src.config import SEMANTIC_GATE_TUNING_PATH
            tuning_results_path = SEMANTIC_GATE_TUNING_PATH
        tuning_path = Path(tuning_results_path)
        if not tuning_path.is_absolute():
            tuning_path = project_root / tuning_path

        # Load tuning results
        self.tuning_results = self._load_tuning_results(tuning_path)
        self.is_hierarchical = self.tuning_results.get("hierarchical", self.tuning_results.get("approach") == "hierarchical_thresholds")

        # Resolve centroids directory
        if centroids_dir is None:
            from src.config import SEMANTIC_GATE_CENTROIDS_DIR
            centroids_dir = SEMANTIC_GATE_CENTROIDS_DIR
        self.centroids_dir = Path(centroids_dir)

        # Load thresholds
        if self.is_hierarchical:
            self.primary_thresholds = self.tuning_results.get("primary_thresholds", {})
            self.secondary_thresholds = self.tuning_results.get("secondary_thresholds", {})
            print(f"[SEMANTIC GATE ONNX] Loaded hierarchical thresholds: {len(self.primary_thresholds)} primary")
        else:
            self.primary_thresholds = self.tuning_results.get("recommendation", {}).get("thresholds", {})
            self.secondary_thresholds = {}
            print(f"[SEMANTIC GATE ONNX] Loaded non-hierarchical thresholds (legacy)")

        # Load ONNX model
        onnx_file = model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        print(f"[SEMANTIC GATE ONNX] Loading ONNX model: {onnx_file}")
        self.session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model_name = "all-MiniLM-L6-v2 (ONNX)"

        # Load centroids
        print(f"[SEMANTIC GATE ONNX] Loading centroids from {self.centroids_dir}")
        self.primary_centroids = self._load_centroids("primary")
        self.secondary_centroids = self._load_centroids("secondary") if self.is_hierarchical else {}

        print(f"[SEMANTIC GATE ONNX] Ready: {len(self.primary_centroids)} primary centroids")

    def _load_tuning_results(self, path: Path) -> dict:
        """Load tuning results from JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Tuning results not found at {path}")
        with open(path, "r") as f:
            results = json.load(f)
        print(f"[SEMANTIC GATE ONNX] Tuning: {results['model_name']}, "
              f"acceptance: {results['global_metrics']['primary']['mean_domain_acceptance'] * 100:.1f}%, "
              f"rejection: {results['global_metrics']['primary']['mean_offtopic_rejection'] * 100:.1f}%")
        return results

    def _load_centroids(self, level: str) -> dict:
        """Load centroids from pickle files."""
        def _normalize(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        centroid_file = self.centroids_dir / f"{level}_centroids.pkl"
        if not centroid_file.exists():
            print(f"[SEMANTIC GATE ONNX] No {level} centroids at {centroid_file}")
            return {}
        try:
            with open(centroid_file, "rb") as f:
                centroids = pickle.load(f)
            if not centroids or not isinstance(centroids, dict):
                return {}
            if level == "primary":
                return {cat: _normalize(arr) for cat, arr in centroids.items()}
            elif level == "secondary":
                return {cat: {subcat: _normalize(arr) for subcat, arr in subcats.items()} for cat, subcats in centroids.items()}
        except Exception as e:
            print(f"[SEMANTIC GATE ONNX] Error loading {level} centroids: {e}")
            return {}
        return {}

    def _encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding using ONNX MiniLM model.

        Applies: tokenize → ONNX inference → mean pooling → L2 normalize.
        Returns shape (1, 384) for MiniLM-L6-v2.
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)

        # Build feed dict
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

        # Run ONNX inference
        outputs = self.session.run(None, feed)
        last_hidden_state = outputs[0]  # (batch, seq_len, hidden_dim)

        # Mean pooling (same as sentence-transformers)
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        mask_expanded = np.broadcast_to(mask_expanded, last_hidden_state.shape)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norms, a_min=1e-9, a_max=None)

        return normalized

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a.flatten(), b.flatten())
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def check_message(
        self, message: str, predicted_category: str, predicted_subcategory: str = None
    ) -> Tuple[bool, float, str, str | None, float | None]:
        """
        Check if message should pass the hierarchical semantic gate.

        Same interface as SemanticGate.check_message().

        Returns:
            Tuple of (should_pass, primary_similarity, best_primary, best_secondary, secondary_similarity)
        """
        # Compute message embedding once
        message_embedding = self._encode(message)

        # STEP 1: Primary Category Check
        best_primary = None
        best_primary_sim = -1.0

        for category, centroid in self.primary_centroids.items():
            similarity = self._cosine_similarity(message_embedding, centroid)
            if similarity > best_primary_sim:
                best_primary_sim = similarity
                best_primary = category

        predicted_category_key = self.CATEGORY_MAPPING.get(predicted_category, predicted_category)
        primary_threshold = self.primary_thresholds.get(predicted_category_key, 0.4)

        if best_primary_sim < primary_threshold:
            return False, float(best_primary_sim), best_primary, None, None

        # STEP 2: Secondary Category Check
        if not self.is_hierarchical:
            return True, float(best_primary_sim), best_primary, None, None

        if predicted_category in self.SKIP_SECONDARY:
            return True, float(best_primary_sim), best_primary, None, None

        if predicted_category_key not in self.secondary_centroids:
            return True, float(best_primary_sim), best_primary, None, None

        if predicted_category_key not in self.secondary_thresholds:
            return True, float(best_primary_sim), best_primary, None, None

        best_secondary = None
        best_secondary_sim = -1.0

        subcategory_centroids = self.secondary_centroids[predicted_category_key]
        for subcat, centroid in subcategory_centroids.items():
            similarity = self._cosine_similarity(message_embedding, centroid)
            if similarity > best_secondary_sim:
                best_secondary_sim = similarity
                best_secondary = subcat

        if predicted_subcategory and predicted_subcategory in self.secondary_thresholds[predicted_category_key]:
            secondary_threshold = self.secondary_thresholds[predicted_category_key][predicted_subcategory]
        elif best_secondary and best_secondary in self.secondary_thresholds[predicted_category_key]:
            secondary_threshold = self.secondary_thresholds[predicted_category_key][best_secondary]
        else:
            secondary_threshold = primary_threshold

        should_pass = best_secondary_sim >= secondary_threshold
        return should_pass, float(best_primary_sim), best_primary, best_secondary, float(best_secondary_sim)

    def get_threshold(self, category: str, subcategory: str = None) -> float:
        """Get threshold for a specific category or subcategory."""
        category_key = self.CATEGORY_MAPPING.get(category, category)

        if subcategory and self.is_hierarchical:
            if category_key in self.secondary_thresholds:
                if subcategory in self.secondary_thresholds[category_key]:
                    return self.secondary_thresholds[category_key][subcategory]

        return self.primary_thresholds.get(category_key, 0.4)

    def get_statistics(self) -> dict:
        """Get semantic gate statistics."""
        stats = {
            "model_name": self.model_name,
            "is_hierarchical": self.is_hierarchical,
            "backend": "onnx",
            "num_primary_categories": len(self.primary_centroids),
            "primary_thresholds": self.primary_thresholds,
            "global_metrics": self.tuning_results["global_metrics"],
        }
        if self.is_hierarchical:
            total_secondary = sum(len(subcats) for subcats in self.secondary_centroids.values())
            stats["num_secondary_categories"] = total_secondary
            stats["secondary_thresholds"] = self.secondary_thresholds
        return stats


# =============================================================================
# Global Singleton
# =============================================================================

_semantic_gate_onnx_instance = None


def get_semantic_gate_onnx(
    model_path: str = None,
    tuning_results_path: str = None,
    centroids_dir: str = None,
    force_reload: bool = False,
) -> SemanticGateONNX:
    """Get or create the ONNX semantic gate singleton."""
    global _semantic_gate_onnx_instance

    if _semantic_gate_onnx_instance is None or force_reload:
        _semantic_gate_onnx_instance = SemanticGateONNX(
            model_path=model_path,
            tuning_results_path=tuning_results_path,
            centroids_dir=centroids_dir,
        )

    return _semantic_gate_onnx_instance
