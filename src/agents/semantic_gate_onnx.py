"""
ONNX-based Hierarchical Semantic Gate - Filters off-topic messages.

Uses ONNX Runtime for MiniLM embeddings + cosine similarity with tuned centroids.
No PyTorch or sentence-transformers required. ~25-40MB RAM.

Two-level filtering:
  - Routing level: Is the message close to any known category centroid?
  - Entity level: Within the predicted context, is it close to an entity centroid?

Data Sources:
  - ONNX Model: {model_path}/model_quantized.onnx (all-MiniLM-L6-v2 sentence embeddings)
  - Tokenizer: {model_path}/tokenizer.json etc.
  - Thresholds: semantic_gate_hierarchical_tuning.json
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
    """

    # Map from classifier output to centroid keys
    CATEGORY_MAPPING = {
        "rag_query": "rag_query",
        "professional": "professional",
        "psychological": "psychological",
        "learning": "learning",
        "social": "social",
        "personal": "personal",
        "chitchat": "chitchat",
        "off_topic": "off_topic",
    }

    # Categories that skip entity-level checking
    SKIP_ENTITY_CHECK = {"rag_query", "chitchat", "off_topic"}

    def __init__(self, model_path: str = None, tuning_results_path: str = None, centroids_dir: str = None):
        """
        Initialize ONNX semantic gate.

        Args:
            model_path: Path to directory with model_quantized.onnx + tokenizer files
                       (the sentence-transformer ONNX model for embeddings)
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
            self.routing_thresholds = self.tuning_results.get("primary_thresholds", {})
            self.entity_thresholds = self.tuning_results.get("secondary_thresholds", {})
            print(f"[SEMANTIC GATE ONNX] Hierarchical thresholds: {len(self.routing_thresholds)} routing, "
                  f"{sum(len(v) for v in self.entity_thresholds.values())} entity")
        else:
            self.routing_thresholds = self.tuning_results.get("recommendation", {}).get("thresholds", {})
            self.entity_thresholds = {}
            print("[SEMANTIC GATE ONNX] Non-hierarchical thresholds (legacy)")

        # Load ONNX embedding model
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
        # primary_centroids.pkl = routing-level (8 categories)
        # secondary_centroids.pkl = entity-level (per context, N entities each)
        print(f"[SEMANTIC GATE ONNX] Loading centroids from {self.centroids_dir}")
        self.routing_centroids = self._load_centroids("primary")
        self.entity_centroids = self._load_centroids("secondary") if self.is_hierarchical else {}

        print(f"[SEMANTIC GATE ONNX] Ready: {len(self.routing_centroids)} routing centroids, "
              f"{sum(len(v) for v in self.entity_centroids.values())} entity centroids")

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

        Applies: tokenize -> ONNX inference -> mean pooling -> L2 normalize.
        Returns shape (1, 384) for MiniLM-L6-v2.
        """
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)

        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))

        outputs = self.session.run(None, feed)
        last_hidden_state = outputs[0]

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        mask_expanded = np.broadcast_to(mask_expanded, last_hidden_state.shape)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norms, a_min=1e-9, a_max=None)

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
        self, message: str, predicted_category: str, predicted_entity: str = None
    ) -> Tuple[bool, float, str, str | None, float | None]:
        """
        Check if message should pass the semantic gate.

        Args:
            message: User message to check
            predicted_category: Route from classifier (e.g. "professional", "rag_query")
            predicted_entity: Entity if applicable (e.g. "professional_aspirations")

        Returns:
            Tuple of (should_pass, routing_similarity, best_routing, best_entity, entity_similarity)
        """
        message_embedding = self._encode(message)

        # =====================================================================
        # STEP 1: Routing-level check
        # =====================================================================
        best_routing = None
        best_routing_sim = -1.0

        for category, centroid in self.routing_centroids.items():
            similarity = self._cosine_similarity(message_embedding, centroid)
            if similarity > best_routing_sim:
                best_routing_sim = similarity
                best_routing = category

        category_key = self.CATEGORY_MAPPING.get(predicted_category, predicted_category)
        routing_threshold = self.routing_thresholds.get(category_key, 0.4)

        if best_routing_sim < routing_threshold:
            return False, float(best_routing_sim), best_routing, None, None

        # =====================================================================
        # STEP 2: Entity-level check (if hierarchical and applicable)
        # =====================================================================
        if not self.is_hierarchical:
            return True, float(best_routing_sim), best_routing, None, None

        if predicted_category in self.SKIP_ENTITY_CHECK:
            return True, float(best_routing_sim), best_routing, None, None

        if category_key not in self.entity_centroids:
            return True, float(best_routing_sim), best_routing, None, None

        if category_key not in self.entity_thresholds:
            return True, float(best_routing_sim), best_routing, None, None

        # Find best matching entity
        best_entity = None
        best_entity_sim = -1.0

        entity_centroids = self.entity_centroids[category_key]
        for entity, centroid in entity_centroids.items():
            similarity = self._cosine_similarity(message_embedding, centroid)
            if similarity > best_entity_sim:
                best_entity_sim = similarity
                best_entity = entity

        # Get entity threshold
        if predicted_entity and predicted_entity in self.entity_thresholds[category_key]:
            entity_threshold = self.entity_thresholds[category_key][predicted_entity]
        elif best_entity and best_entity in self.entity_thresholds[category_key]:
            entity_threshold = self.entity_thresholds[category_key][best_entity]
        else:
            entity_threshold = routing_threshold

        should_pass = best_entity_sim >= entity_threshold
        return should_pass, float(best_routing_sim), best_routing, best_entity, float(best_entity_sim)

    def get_threshold(self, category: str, entity: str = None) -> float:
        """Get threshold for a specific category or entity."""
        category_key = self.CATEGORY_MAPPING.get(category, category)

        if entity and self.is_hierarchical:
            if category_key in self.entity_thresholds:
                if entity in self.entity_thresholds[category_key]:
                    return self.entity_thresholds[category_key][entity]

        return self.routing_thresholds.get(category_key, 0.4)

    def get_statistics(self) -> dict:
        """Get semantic gate statistics."""
        stats = {
            "model_name": self.model_name,
            "is_hierarchical": self.is_hierarchical,
            "backend": "onnx",
            "num_routing_categories": len(self.routing_centroids),
            "routing_thresholds": self.routing_thresholds,
            "global_metrics": self.tuning_results["global_metrics"],
        }
        if self.is_hierarchical:
            total_entity = sum(len(v) for v in self.entity_centroids.values())
            stats["num_entity_categories"] = total_entity
            stats["entity_thresholds"] = self.entity_thresholds
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
