"""
Hierarchical Semantic Gate - Two-Level Filtering (ONNX Runtime)

Filters off-topic messages using hierarchical thresholds:
- Primary level: Category thresholds (rag_query, professional, etc.)
- Secondary level: Subcategory thresholds (skills, experience, etc.)

Flow:
1. Compute message embedding via ONNX model (MiniLM quantized)
2. Primary check: Compare to primary category centroids
3. Secondary check (if applicable): Compare to subcategory centroids
4. Block if below threshold at either level

Data Sources (all from {ONNX_MODELS_PATH}/semantic_gate/):
- model_quantized.onnx + tokenizer files (MiniLM-L6-v2)
- primary_centroids.pkl, secondary_centroids.pkl
- semantic_gate_hierarchical_tuning.json
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import ClassVar, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _mean_pooling(hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean Pooling — same logic as sentence-transformers default pooling."""
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(hidden_state * mask, axis=1)
    sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings (same as sentence-transformers)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two 2D arrays. Returns shape (len(a), len(b))."""
    a_norm = _l2_normalize(a)
    b_norm = _l2_normalize(b)
    return np.dot(a_norm, b_norm.T)


class SemanticGate:
    """
    Hierarchical semantic gate for filtering off-topic messages.

    Uses ONNX Runtime for embeddings and cosine similarity with two-level thresholds:
    - Primary: Category-level thresholds
    - Secondary: Subcategory-level thresholds
    """

    # Default similarity threshold when a category has no tuned value
    DEFAULT_THRESHOLD: ClassVar[float] = 0.4

    # Mapping from MessageCategory enum values to tuning JSON keys
    CATEGORY_MAPPING: ClassVar[dict[str, str]] = {
        "rag_query": "rag_query",
        "professional": "professional",
        "psychological": "psychological",
        "learning": "learning",
        "social": "social",
        "emotional": "psychological",
        "aspirational": "professional",
        "personal": "personal",
        "chitchat": "chitchat",
        "meta": "meta",
        "off_topic": "off_topic",
    }

    # Categories that skip secondary classification
    SKIP_SECONDARY: ClassVar[set[str]] = {"rag_query", "chitchat", "meta", "off_topic"}

    def __init__(
        self,
        tuning_results_path: str | None = None,
        centroids_dir: str | None = None,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize hierarchical semantic gate.

        Args:
            tuning_results_path: Path to semantic_gate_hierarchical_tuning.json.
                                 Defaults to {centroids_dir}/semantic_gate_hierarchical_tuning.json
            centroids_dir: Directory containing ONNX model, tokenizer, centroids, and tuning JSON.
                          Defaults to {ONNX_MODELS_PATH}/semantic_gate/
            model_name: Original model name (for logging only)
        """
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Default centroids directory
        project_root = Path(__file__).parent.parent.parent
        if centroids_dir is None:
            from src.config import ONNX_MODELS_PATH
            centroids_dir = Path(ONNX_MODELS_PATH) / "semantic_gate"
        else:
            centroids_dir = Path(centroids_dir)

        self.centroids_dir = centroids_dir

        # Default tuning results path: look in centroids_dir first, then training/results/
        if tuning_results_path is None:
            candidate = centroids_dir / "semantic_gate_hierarchical_tuning.json"
            if candidate.exists():
                tuning_results_path = candidate
            else:
                tuning_results_path = project_root / "training" / "results" / "semantic_gate_hierarchical_tuning.json"

        # Load tuning results (hierarchical)
        self.tuning_results = self._load_tuning_results(tuning_results_path)
        self.is_hierarchical = self.tuning_results.get("hierarchical", self.tuning_results.get("approach") == "hierarchical_thresholds")

        # Load thresholds
        if self.is_hierarchical:
            self.primary_thresholds = self.tuning_results.get("primary_thresholds", {})
            self.secondary_thresholds = self.tuning_results.get("secondary_thresholds", {})
            logger.info("Loaded hierarchical thresholds:")
            logger.info(f"Primary categories: {len(self.primary_thresholds)}")
            total_secondary = sum(len(subcats) for subcats in self.secondary_thresholds.values())
            logger.info(f"Secondary categories: {total_secondary}")
        else:
            # Fallback to old format (backwards compatibility)
            self.primary_thresholds = self.tuning_results.get("recommendation", {}).get("thresholds", {})
            self.secondary_thresholds = {}
            logger.info("Loaded non-hierarchical thresholds (legacy format)")
            logger.info(f"Primary categories: {len(self.primary_thresholds)}")

        # Load ONNX model + tokenizer from centroids_dir
        onnx_path = centroids_dir / "model_quantized.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        logger.info(f"Loading ONNX semantic gate model from {centroids_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(centroids_dir))
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.model_name = model_name
        logger.info(f"Loaded ONNX model: {model_name} (quantized)")

        # Load centroids from pickle files
        logger.info(f"Loading centroids from {centroids_dir}...")
        self.primary_centroids = self._load_centroids_from_models("primary")
        self.secondary_centroids = self._load_centroids_from_models("secondary") if self.is_hierarchical else {}

        logger.info("Initialization complete")
        logger.info(f"Primary centroids: {len(self.primary_centroids)}")
        if self.is_hierarchical:
            total_sec_centroids = sum(len(subcats) for subcats in self.secondary_centroids.values())
            logger.info(f"Secondary centroids: {total_sec_centroids}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings using ONNX model + mean pooling."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        outputs = self.session.run(None, {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
            "token_type_ids": tokens.get(
                "token_type_ids",
                np.zeros_like(tokens["input_ids"]),
            ).astype(np.int64),
        })
        hidden_state = outputs[0]  # (batch, seq_len, 384)
        embeddings = _mean_pooling(hidden_state, tokens["attention_mask"])
        return _l2_normalize(embeddings)

    def _load_tuning_results(self, path: str) -> dict:
        """Load tuning results from JSON file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Semantic gate tuning results not found at {path}. Run: python training/scripts/tune_semantic_gate.py")

        with open(path, "r") as f:
            results = json.load(f)

        logger.info(f"Loaded tuning results from {path}")
        logger.info(f"Model: {results['model_name']}")
        logger.info(f"Mean domain acceptance: {results['global_metrics']['primary']['mean_domain_acceptance'] * 100:.2f}%")
        logger.info(f"Mean off-topic rejection: {results['global_metrics']['primary']['mean_offtopic_rejection'] * 100:.2f}%")

        return results

    def _load_centroids_from_models(self, level: str):
        """
        Load centroids from pickle files in centroids_dir.

        Args:
            level: "primary" or "secondary"

        Returns:
            For primary: Dict mapping category -> centroid array (shape 1, dim)
            For secondary: Dict mapping category -> subcategory -> centroid array
        """
        def _normalize(arr):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        if level in ("primary", "secondary"):
            centroid_file = self.centroids_dir / f"{level}_centroids.pkl"
            if not centroid_file.exists():
                logger.info(f"No {level} centroids at {centroid_file}")
                return {}
            try:
                with open(centroid_file, "rb") as f:
                    centroids = pickle.load(f)
                if not centroids or not isinstance(centroids, dict):
                    logger.info(f"Invalid {level} centroids format")
                    return {}
                if level == "primary":
                    return {cat: _normalize(arr) for cat, arr in centroids.items()}
                elif level == "secondary":
                    return {cat: {subcat: _normalize(arr) for subcat, arr in subcats.items()} for cat, subcats in centroids.items()}
            except Exception as e:
                logger.info(f"Error loading {level} centroids: {e}")
                return {}

        return {}

    def check_message(
        self,
        message: str,
        predicted_category: str,
        predicted_subcategory: str | None = None
    ) -> Tuple[bool, float, str, str | None, float | None]:
        """
        Check if message should pass the hierarchical semantic gate.

        Args:
            message: User message to check
            predicted_category: Primary category predicted by intent classifier (e.g., "rag_query")
            predicted_subcategory: Secondary category if applicable (e.g., "skills")

        Returns:
            Tuple of (should_pass, primary_similarity, best_primary, best_secondary, secondary_similarity)
        """
        # Compute message embedding once
        message_embedding = self._encode([message])

        # =====================================================================
        # STEP 1: Primary Category Check
        # =====================================================================
        best_primary = None
        best_primary_sim = -1

        for category, centroid in self.primary_centroids.items():
            similarity = _cosine_similarity(message_embedding, centroid)[0][0]
            if similarity > best_primary_sim:
                best_primary_sim = similarity
                best_primary = category

        # Map predicted category to tuning key format
        predicted_category_key = self.CATEGORY_MAPPING.get(predicted_category, predicted_category)

        # Get primary threshold
        primary_threshold = self.primary_thresholds.get(predicted_category_key, self.DEFAULT_THRESHOLD)

        # Check primary threshold
        if best_primary_sim < primary_threshold:
            # Failed primary check
            return False, float(best_primary_sim), best_primary, None, None

        # =====================================================================
        # STEP 2: Secondary Category Check (if hierarchical and applicable)
        # =====================================================================
        if not self.is_hierarchical:
            return True, float(best_primary_sim), best_primary, None, None

        if predicted_category in self.SKIP_SECONDARY:
            return True, float(best_primary_sim), best_primary, None, None

        if predicted_category_key not in self.secondary_centroids:
            return True, float(best_primary_sim), best_primary, None, None

        if predicted_category_key not in self.secondary_thresholds:
            return True, float(best_primary_sim), best_primary, None, None

        # Find best matching subcategory
        best_secondary = None
        best_secondary_sim = -1

        subcategory_centroids = self.secondary_centroids[predicted_category_key]
        for subcat, centroid in subcategory_centroids.items():
            similarity = _cosine_similarity(message_embedding, centroid)[0][0]
            if similarity > best_secondary_sim:
                best_secondary_sim = similarity
                best_secondary = subcat

        # Get secondary threshold
        if predicted_subcategory and predicted_subcategory in self.secondary_thresholds[predicted_category_key]:
            secondary_threshold = self.secondary_thresholds[predicted_category_key][predicted_subcategory]
        elif best_secondary and best_secondary in self.secondary_thresholds[predicted_category_key]:
            secondary_threshold = self.secondary_thresholds[predicted_category_key][best_secondary]
        else:
            secondary_threshold = primary_threshold

        # Check secondary threshold
        should_pass = best_secondary_sim >= secondary_threshold

        return should_pass, float(best_primary_sim), best_primary, best_secondary, float(best_secondary_sim)

    def get_threshold(self, category: str, subcategory: str | None = None) -> float:
        """Get threshold for a specific category or subcategory."""
        category_key = self.CATEGORY_MAPPING.get(category, category)

        if subcategory and self.is_hierarchical:
            if category_key in self.secondary_thresholds:
                if subcategory in self.secondary_thresholds[category_key]:
                    return self.secondary_thresholds[category_key][subcategory]

        return self.primary_thresholds.get(category_key, self.DEFAULT_THRESHOLD)

    def get_statistics(self) -> dict:
        """Get semantic gate statistics"""
        stats = {
            "model_name": self.model_name,
            "is_hierarchical": self.is_hierarchical,
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
# Global Singleton (Lazy-loaded)
# =============================================================================

_semantic_gate_instance = None


def get_semantic_gate(
    centroids_dir: str | None = None,
    tuning_results_path: str | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    force_reload: bool = False
) -> SemanticGate:
    """
    Get or create the global semantic gate instance (singleton pattern).

    Args:
        centroids_dir: Directory containing ONNX model, tokenizer, centroids, and tuning JSON
        tuning_results_path: Path to tuning results JSON (defaults to centroids_dir/)
        model_name: Model name for logging
        force_reload: If True, recreate the semantic gate

    Returns:
        SemanticGate instance
    """
    global _semantic_gate_instance

    if _semantic_gate_instance is None or force_reload:
        _semantic_gate_instance = SemanticGate(
            centroids_dir=centroids_dir,
            tuning_results_path=tuning_results_path,
            model_name=model_name
        )

    return _semantic_gate_instance
