"""
Hierarchical ONNX-based Intent Classifier.

4-level classification pipeline using ONNX Runtime:
  1. Routing (softmax 8 classes) → determines context or non-context type
  2. Contexts (softmax 5 classes) → confirms which context
  3. Entities (softmax N classes per context) → which entities within context
  4. Sub-entities (sigmoid multi-label per entity) → which sub-entities (multiple active)

All models loaded from disk as quantized ONNX. No PyTorch required.

Directory structure expected:
    {model_path}/
    ├── hierarchy_metadata.json
    ├── routing/          (model_quantized.onnx, config.json, label_mappings.json, tokenizer)
    ├── contexts/         (model_quantized.onnx, ...)
    ├── professional/
    │   ├── entities/     (model_quantized.onnx, ...)
    │   ├── current_position/      (model_quantized.onnx, ... multi-label)
    │   ├── professional_aspirations/
    │   └── ...
    ├── learning/
    │   ├── entities/
    │   └── ...
    └── ...
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# =============================================================================
# Data types
# =============================================================================

@dataclass
class SubEntityResult:
    """Result of sub-entity multi-label classification."""
    entity: str
    sub_entities: list[str]
    probabilities: dict[str, float]


@dataclass
class ContextResult:
    """Result for a single detected context path."""
    context: str                         # e.g. "professional"
    context_confidence: float
    entities: list[SubEntityResult] = field(default_factory=list)  # detected entities + their sub-entities


@dataclass
class HierarchicalClassification:
    """Full hierarchical classification result."""
    # Level 0: Routing
    route: str                           # e.g. "professional", "rag_query", "chitchat"
    route_confidence: float
    is_context: bool                     # True if route is one of 5 contexts

    # All detected context paths (multiple contexts possible)
    contexts: list[ContextResult] = field(default_factory=list)

    # All route probabilities for secondary analysis
    route_probabilities: dict[str, float] = field(default_factory=dict)

    # Reasoning string
    reasoning: str = ""


# =============================================================================
# Single ONNX model wrapper
# =============================================================================

class ONNXModel:
    """Wrapper for a single ONNX classification model."""

    def __init__(self, model_dir: Path, name: str = ""):
        self.name = name
        self.model_dir = model_dir

        # Find ONNX model file (prefer quantized)
        onnx_file = model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        self.session = ort.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Load label mappings: prefer label_mappings.json (explicit export) over config.json
        # config.json often has default LABEL_0/LABEL_1 from HuggingFace
        config_file = model_dir / "config.json"
        with open(config_file) as f:
            config = json.load(f)
        self.problem_type = config.get("problem_type", "single_label_classification")

        # Start with config.json id2label as fallback
        self.id2label = config.get("id2label", {})

        # label_mappings.json always takes priority when it exists
        label_mappings_file = model_dir / "label_mappings.json"
        # Default threshold slightly above 0.5 so sigmoid "no signal" (~0.50) is rejected
        self.threshold = 0.55
        if label_mappings_file.exists():
            with open(label_mappings_file) as f:
                label_config = json.load(f)
            if label_config.get("problem_type") == "multi_label_classification":
                self.problem_type = "multi_label_classification"
            self.threshold = label_config.get("threshold", 0.5)
            # Check for nested "label_mappings" key or flat {0: "name"} format
            if "label_mappings" in label_config:
                self.id2label = label_config["label_mappings"]
            elif all(k.isdigit() for k in label_config if k not in ("problem_type", "threshold")):
                # Flat format: {"0": "professional", "1": "learning", ...}
                self.id2label = {k: v for k, v in label_config.items()
                                 if k not in ("problem_type", "threshold")}

        self.num_labels = len(self.id2label)

        self.is_multilabel = self.problem_type == "multi_label_classification"

    def _tokenize(self, message: str) -> dict:
        """Tokenize and build feed dict."""
        inputs = self.tokenizer(
            message, return_tensors="np", padding=True, truncation=True, max_length=128
        )
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in self.input_names:
            tid = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
            feed["token_type_ids"] = tid.astype(np.int64)
        return feed

    def predict_single_label(self, message: str) -> tuple[str, float, dict[str, float]]:
        """
        Single-label prediction (softmax).

        Returns:
            (best_label, confidence, all_probabilities)
        """
        feed = self._tokenize(message)
        outputs = self.session.run(None, feed)
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        best_idx = int(np.argmax(probs))
        best_label = self.id2label.get(str(best_idx), f"label_{best_idx}")
        confidence = float(probs[best_idx])

        all_probs = {
            self.id2label.get(str(i), f"label_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        return best_label, confidence, all_probs

    def predict_multi_label(self, message: str, threshold: float = None) -> tuple[list[str], dict[str, float]]:
        """
        Multi-label prediction (sigmoid).

        Returns:
            (active_labels, all_probabilities)
        """
        if threshold is None:
            threshold = self.threshold

        feed = self._tokenize(message)
        outputs = self.session.run(None, feed)
        logits = outputs[0][0]

        # Sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))

        all_probs = {
            self.id2label.get(str(i), f"label_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        active = [label for label, prob in all_probs.items() if prob >= threshold]

        return active, all_probs


# =============================================================================
# Hierarchical classifier
# =============================================================================

CONTEXT_TYPES = {"professional", "learning", "social", "psychological", "personal"}


class HierarchicalONNXClassifier:
    """
    Hierarchical 4-level ONNX classifier with multi-path detection.

    For each message, detects MULTIPLE contexts, entities, and sub-entities
    using probability thresholds at every level (not just argmax).

    Pipeline:
      routing (top-N contexts above threshold)
        → for each context: entities (top-N above threshold)
          → for each entity: sub-entities (sigmoid multi-label)
    """

    # Entity model threshold: each context has ~6-17 entities (softmax).
    # Random uniform = ~10-17%. A real signal gives the top entity >50%.
    # 30% filters noise while keeping legitimate detections like social (34%).
    # This is the ONLY filter — routing doesn't limit which contexts to explore.
    ENTITY_THRESHOLD = 0.30

    def __init__(self, model_path: str):
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"[HIERARCHICAL ONNX] Loading models from {model_dir}...")

        # Load hierarchy metadata
        metadata_file = model_dir / "hierarchy_metadata.json"
        self.metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)

        # Level 0: Routing
        self.routing_model = self._load_model(model_dir / "routing", "routing")

        # Level 1: Contexts
        self.contexts_model = self._load_model(model_dir / "contexts", "contexts")

        # Level 2: Entity models (one per context)
        self.entity_models: dict[str, ONNXModel] = {}
        for ctx in CONTEXT_TYPES:
            entity_dir = model_dir / ctx / "entities"
            if entity_dir.exists():
                model = self._load_model(entity_dir, f"{ctx}/entities")
                if model:
                    self.entity_models[ctx] = model

        # Level 3: Sub-entity models skipped — LLM extraction handles sub-entities
        # The classifier only needs to identify context + entity (2 levels).
        # Sub-entities are looked up from CONTEXT_REGISTRY and extracted by the LLM.
        self.sub_entity_models: dict[str, dict[str, ONNXModel]] = {}

        # Summary
        n_entity = len(self.entity_models)
        print(f"[HIERARCHICAL ONNX] Loaded: routing + contexts + {n_entity} entity models (sub-entities handled by LLM)")

    def _load_model(self, model_dir: Path, name: str) -> Optional[ONNXModel]:
        """Load a single ONNX model, returning None if not found."""
        try:
            model = ONNXModel(model_dir, name)
            print(f"  [{name}] {model.num_labels} labels ({'multi-label' if model.is_multilabel else 'single-label'})")
            return model
        except FileNotFoundError:
            print(f"  [{name}] not found, skipping")
            return None

    def _get_entities_above_threshold(self, entity_probs: dict[str, float]) -> list[tuple[str, float]]:
        """
        Get all entities with probability above threshold.

        Returns list of (entity, probability) sorted by probability descending.
        """
        entities = []
        for label, prob in entity_probs.items():
            if prob >= self.ENTITY_THRESHOLD:
                entities.append((label, prob))
        entities.sort(key=lambda x: x[1], reverse=True)
        return entities

    async def classify(self, message: str) -> HierarchicalClassification:
        """
        Run full hierarchical classification with multi-path detection.

        Flow:
            1. Routing → get ALL contexts above threshold (not just top-1)
            2. For each context → get ALL entities above threshold
            3. For each entity → get sub-entities (sigmoid multi-label)

        Example result for "Quiero ser CEO de Apple, gano 100k y sé Python":
            contexts:
              - professional (0.65):
                  - professional_aspirations → [dream_roles, compensation_expectations]
                  - current_position → [compensation]
              - learning (0.25):
                  - current_skills → [skills]
        """
        # =====================================================================
        # STEP 1: Routing (8 classes) → find ALL relevant contexts
        # =====================================================================
        if not self.routing_model:
            return HierarchicalClassification(
                route="chitchat", route_confidence=0.0, is_context=False,
                reasoning="No routing model loaded",
            )

        route, route_conf, route_probs = self.routing_model.predict_single_label(message)
        is_context = route in CONTEXT_TYPES

        if not is_context:
            # Non-context route (rag_query, chitchat, off_topic) → no extraction
            return HierarchicalClassification(
                route=route,
                route_confidence=route_conf,
                is_context=False,
                route_probabilities=route_probs,
                reasoning=f"Routing: {route} ({route_conf:.1%})",
            )

        # =================================================================
        # Route IS a context → run ALL 5 entity models (ONNX, free).
        # The routing model is softmax and concentrates ~97% on one context,
        # so we DON'T use routing probabilities to filter contexts.
        # Instead, each entity model decides if its context is relevant
        # based on ENTITY_THRESHOLD (40%). Only those trigger LLM calls.
        # =================================================================
        detected_contexts = [
            (ctx, 1.0) for ctx in self.entity_models
        ]
        print(f"[HIERARCHICAL ONNX] Running all {len(detected_contexts)} entity models")

        # =====================================================================
        # STEP 2 & 3: For each context → entities → sub-entities
        # =====================================================================
        context_results = []
        reasoning_parts = [f"Routing: {route} ({route_conf:.1%})"]

        for ctx, ctx_conf in detected_contexts:
            # Get entities for this context
            entity_results = []

            if ctx in self.entity_models:
                _, _, entity_probs = self.entity_models[ctx].predict_single_label(message)
                top_entity = max(entity_probs.items(), key=lambda x: x[1])
                print(f"[HIERARCHICAL ONNX]   {ctx}: top={top_entity[0]} ({top_entity[1]:.1%})")
                detected_entities = self._get_entities_above_threshold(entity_probs)

                for entity, entity_conf in detected_entities:
                    # Sub-entities are resolved downstream by the LLM extraction node
                    # using CONTEXT_REGISTRY lookups — no Level 3 ONNX model needed.
                    entity_results.append(SubEntityResult(
                        entity=entity,
                        sub_entities=[],
                        probabilities={},
                    ))
                    reasoning_parts.append(f"{ctx}.{entity} ({entity_conf:.0%})")

            if entity_results:
                context_results.append(ContextResult(
                    context=ctx,
                    context_confidence=ctx_conf,
                    entities=entity_results,
                ))

        return HierarchicalClassification(
            route=route,
            route_confidence=route_conf,
            is_context=bool(context_results),
            contexts=context_results,
            route_probabilities=route_probs,
            reasoning=" | ".join(reasoning_parts),
        )


# =============================================================================
# Singleton
# =============================================================================

_classifier_instance: Optional[HierarchicalONNXClassifier] = None


def get_hierarchical_classifier(model_path: str = "") -> HierarchicalONNXClassifier:
    """Get or create the hierarchical ONNX classifier singleton."""
    global _classifier_instance
    if _classifier_instance is None:
        if not model_path:
            from src.config import HIERARCHICAL_MODEL_PATH
            model_path = HIERARCHICAL_MODEL_PATH
        _classifier_instance = HierarchicalONNXClassifier(model_path)
    return _classifier_instance
