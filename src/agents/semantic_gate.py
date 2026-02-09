"""
Hierarchical Semantic Gate - Two-Level Filtering

Filters off-topic messages using hierarchical thresholds:
- Primary level: Category thresholds (rag_query, professional, etc.)
- Secondary level: Subcategory thresholds (skills, experience, etc.)

Flow:
1. Compute message embedding
2. Primary check: Compare to primary category centroids
3. Secondary check (if applicable): Compare to subcategory centroids
4. Block if below threshold at either level

Data Sources:
- Thresholds: training/results/semantic_gate_hierarchical_tuning.json
- Centroids: from hierarchical model path (INTENT_CLASSIFIER_MODEL_PATH or tuning model_path)
  - Primary: {model_path}/primary/final/centroids.pkl
  - Secondary: {model_path}/secondary/{category}/final/centroids.pkl per category
"""

import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

# Lazy imports for optional dependencies
_sentence_transformer = None
_cosine_similarity = None


def _ensure_dependencies():
    """Ensure required dependencies are imported (lazy loading)"""
    global _sentence_transformer, _cosine_similarity

    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            _sentence_transformer = SentenceTransformer
            _cosine_similarity = cosine_similarity
        except ImportError:
            raise ImportError(
                "Semantic gate requires sentence-transformers and scikit-learn. Install with: pip install sentence-transformers scikit-learn"
            )


class SemanticGate:
    """
    Hierarchical semantic gate for filtering off-topic messages.

    Uses SentenceTransformer embeddings and cosine similarity with two-level thresholds:
    - Primary: Category-level thresholds
    - Secondary: Subcategory-level thresholds
    """

    # Mapping from MessageCategory enum values to tuning JSON keys
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

    # Categories that skip secondary classification
    SKIP_SECONDARY = {"rag_query", "chitchat", "off_topic"}

    def __init__(self, tuning_results_path: str = None, centroids_dir: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize hierarchical semantic gate.

        Args:
            tuning_results_path: Path to semantic_gate_hierarchical_tuning.json
            centroids_dir: Directory containing centroid pickle files (primary_centroids.pkl, secondary_centroids.pkl)
                          Defaults to {INTENT_CLASSIFIER_MODEL_PATH}/semantic_gate/
            model_name: SentenceTransformer model name (must match tuning)
        """
        _ensure_dependencies()

        # Default paths (relative to project root)
        project_root = Path(__file__).parent.parent.parent
        if tuning_results_path is None:
            tuning_results_path = project_root / "training" / "results" / "semantic_gate_hierarchical_tuning.json"

        # Load tuning results (hierarchical)
        self.tuning_results = self._load_tuning_results(tuning_results_path)
        self.is_hierarchical = self.tuning_results.get("hierarchical", self.tuning_results.get("approach") == "hierarchical_thresholds")

        # Determine centroids directory
        # Expected: {model_path}/semantic_gate/ containing primary_centroids.pkl and secondary_centroids.pkl
        if centroids_dir is None:
            # Try tuning results model_path, then INTENT_CLASSIFIER_MODEL_PATH
            model_path = self.tuning_results.get("model_path")
            if model_path:
                centroids_dir = Path(model_path) / "semantic_gate"
            else:
                try:
                    from src.config import INTENT_CLASSIFIER_MODEL_PATH

                    centroids_dir = Path(INTENT_CLASSIFIER_MODEL_PATH) / "semantic_gate"
                except ImportError:
                    centroids_dir = project_root / "training" / "models" / "semantic_gate"
        else:
            centroids_dir = Path(centroids_dir)

        self.centroids_dir = centroids_dir

        # Load thresholds
        if self.is_hierarchical:
            self.primary_thresholds = self.tuning_results.get("primary_thresholds", {})
            self.secondary_thresholds = self.tuning_results.get("secondary_thresholds", {})
            print("[SEMANTIC GATE] Loaded hierarchical thresholds:")
            print(f"  Primary categories: {len(self.primary_thresholds)}")
            total_secondary = sum(len(subcats) for subcats in self.secondary_thresholds.values())
            print(f"  Secondary categories: {total_secondary}")
        else:
            # Fallback to old format (backwards compatibility)
            self.primary_thresholds = self.tuning_results.get("recommendation", {}).get("thresholds", {})
            self.secondary_thresholds = {}
            print("[SEMANTIC GATE] Loaded non-hierarchical thresholds (legacy format)")
            print(f"  Primary categories: {len(self.primary_thresholds)}")

        # Initialize embedding model
        print(f"[SEMANTIC GATE] Loading SentenceTransformer: {model_name}...")
        self.model = _sentence_transformer(model_name)
        self.model_name = model_name

        # Load centroids from models directory (pickle files)
        print(f"[SEMANTIC GATE] Loading centroids from {centroids_dir}...")
        self.primary_centroids = self._load_centroids_from_models("primary")
        self.secondary_centroids = self._load_centroids_from_models("secondary") if self.is_hierarchical else {}

        print("[SEMANTIC GATE] Initialization complete")
        print(f"  Primary centroids: {len(self.primary_centroids)}")
        if self.is_hierarchical:
            total_sec_centroids = sum(len(subcats) for subcats in self.secondary_centroids.values())
            print(f"  Secondary centroids: {total_sec_centroids}")

    def _load_tuning_results(self, path: str) -> dict:
        """Load tuning results from JSON file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Semantic gate tuning results not found at {path}. Run: python training/scripts/tune_semantic_gate.py")

        with open(path, "r") as f:
            results = json.load(f)

        print(f"[SEMANTIC GATE] Loaded tuning results from {path}")
        print(f"  Model: {results['model_name']}")
        print(f"  Mean domain acceptance: {results['global_metrics']['primary']['mean_domain_acceptance'] * 100:.2f}%")
        print(f"  Mean off-topic rejection: {results['global_metrics']['primary']['mean_offtopic_rejection'] * 100:.2f}%")

        return results

    def _load_centroids_from_models(self, level: str):
        """
        Load centroids from hierarchical model directory.

        Paths:
        - Primary: centroids_dir/primary/final/centroids.pkl
        - Secondary: centroids_dir/secondary/{category}/final/centroids.pkl (one .pkl per category)

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
                print(f"[SEMANTIC GATE] No {level} centroids at {centroid_file}")
                return {}
            try:
                with open(centroid_file, "rb") as f:
                    centroids = pickle.load(f)
                if not centroids or not isinstance(centroids, dict):
                    print(f"[SEMANTIC GATE] Invalid {level} centroids format")
                    return {}
                if level == "primary":
                    return {cat: _normalize(arr) for cat, arr in centroids.items()}
                elif level == "secondary":
                    return {cat: {subcat: _normalize(arr) for subcat, arr in subcats.items()} for cat, subcats in centroids.items()}
            except Exception as e:
                print(f"[SEMANTIC GATE] Error loading {level} centroids: {e}")
                return {}

        return {}

    def check_message(
        self, message: str, predicted_category: str, predicted_subcategory: str = None
    ) -> Tuple[bool, float, str, str | None, float | None]:
        """
        Check if message should pass the hierarchical semantic gate.

        Args:
            message: User message to check
            predicted_category: Primary category predicted by intent classifier (e.g., "rag_query")
            predicted_subcategory: Secondary category if applicable (e.g., "skills")

        Returns:
            Tuple of (should_pass, primary_similarity, best_primary, best_secondary, secondary_similarity)
            - should_pass: True if message passes both levels, False if blocked at either level
            - primary_similarity: Similarity to best matching primary category
            - best_primary: Best matching primary category (tuning key format)
            - best_secondary: Best matching secondary category (or None if N/A)
            - secondary_similarity: Similarity to best matching secondary category (or None if N/A)
        """
        # Compute message embedding once
        message_embedding = self.model.encode([message])

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
        primary_threshold = self.primary_thresholds.get(predicted_category_key, 0.4)

        # Check primary threshold
        if best_primary_sim < primary_threshold:
            # Failed primary check
            return False, float(best_primary_sim), best_primary, None, None

        # =====================================================================
        # STEP 2: Secondary Category Check (if hierarchical and applicable)
        # =====================================================================
        if not self.is_hierarchical:
            # Non-hierarchical mode: only primary check
            return True, float(best_primary_sim), best_primary, None, None

        # Check if category has secondary classification
        if predicted_category in self.SKIP_SECONDARY:
            # Categories that skip secondary (rag_query, chitchat, off_topic)
            return True, float(best_primary_sim), best_primary, None, None

        # Check if we have secondary centroids and thresholds for this category
        if predicted_category_key not in self.secondary_centroids:
            # No secondary data for this category
            return True, float(best_primary_sim), best_primary, None, None

        if predicted_category_key not in self.secondary_thresholds:
            # No secondary thresholds for this category
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
            # Use predicted subcategory threshold
            secondary_threshold = self.secondary_thresholds[predicted_category_key][predicted_subcategory]
        elif best_secondary and best_secondary in self.secondary_thresholds[predicted_category_key]:
            # Use best match threshold
            secondary_threshold = self.secondary_thresholds[predicted_category_key][best_secondary]
        else:
            # Fallback: use primary threshold
            secondary_threshold = primary_threshold

        # Check secondary threshold
        should_pass = best_secondary_sim >= secondary_threshold

        return should_pass, float(best_primary_sim), best_primary, best_secondary, float(best_secondary_sim)

    def get_threshold(self, category: str, subcategory: str = None) -> float:
        """
        Get threshold for a specific category or subcategory.

        Args:
            category: Category name (MessageCategory enum value like "rag_query")
            subcategory: Optional subcategory name (e.g., "skills")

        Returns:
            Threshold for that category/subcategory
        """
        # Map to tuning key format
        category_key = self.CATEGORY_MAPPING.get(category, category)

        if subcategory and self.is_hierarchical:
            # Try to get secondary threshold
            if category_key in self.secondary_thresholds:
                if subcategory in self.secondary_thresholds[category_key]:
                    return self.secondary_thresholds[category_key][subcategory]

        # Get primary threshold
        return self.primary_thresholds.get(category_key, 0.4)

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


def get_semantic_gate(centroids_dir: str = None, tuning_results_path: str = None, force_reload: bool = False) -> SemanticGate:
    """
    Get or create the global semantic gate instance (singleton pattern).

    Args:
        force_reload: If True, recreate the semantic gate

    Returns:
        SemanticGate instance
    """
    global _semantic_gate_instance

    if _semantic_gate_instance is None or force_reload:
        _semantic_gate_instance = SemanticGate(centroids_dir=centroids_dir, tuning_results_path=tuning_results_path)

    return _semantic_gate_instance
