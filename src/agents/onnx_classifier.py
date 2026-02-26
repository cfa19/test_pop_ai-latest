"""
ONNX Token Classifier for Hierarchical Intent Classification.

Replaces the PyTorch SpanClassifier + DistilBertIntentClassifier with
ONNX Runtime inference. Each token gets a label; consecutive tokens
with the same label are merged into spans.

Primary classifier: assigns context labels (professional, learning, etc.)
Secondary classifiers: assign entity labels per context (work_history, etc.)

Directory structure expected:
    {models_dir}/
    ├── primary/            (model_quantized.onnx, tokenizer.json, label_maps.json)
    └── secondary/
        ├── professional/   (model_quantized.onnx, tokenizer.json, label_maps.json)
        ├── learning/
        ├── personal/
        ├── psychological/
        └── social/
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SECONDARY_CONTEXTS = ["professional", "learning", "personal", "psychological", "social"]


@dataclass
class Span:
    text: str
    label: str
    start: int
    end: int


@dataclass
class HierarchicalResult:
    """Result from hierarchical token classification."""
    # Primary spans (context-level)
    primary_spans: list[Span] = field(default_factory=list)
    # Active contexts found
    active_contexts: list[str] = field(default_factory=list)
    # Entity spans per context: {"professional": [Span(label="work_history", ...), ...]}
    entity_spans: dict[str, list[Span]] = field(default_factory=dict)


class ONNXTokenClassifier:
    """Single ONNX token classifier (primary or secondary)."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)

        # Load ONNX session
        onnx_path = model_dir / "model_quantized.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Load label maps
        label_maps_path = model_dir / "label_maps.json"
        with open(label_maps_path) as f:
            maps = json.load(f)
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}

        logger.info(f"Loaded ONNX model from {model_dir}: labels={list(self.id2label.values())}")

    def classify_spans(self, text: str) -> list[Span]:
        """
        Run token classification and return labeled text spans.

        Tokenizes text, runs ONNX inference, takes argmax per token,
        then merges consecutive tokens with the same non-O label into spans.
        """
        encoding = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        logits = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })[0]

        pred_ids = np.argmax(logits[0], axis=-1).tolist()

        spans: list[Span] = []
        current: Span | None = None

        for pred_id, (char_start, char_end) in zip(pred_ids, offset_mapping):
            if char_start == char_end == 0:  # special token
                continue

            label = self.id2label[pred_id]
            parsed = self._parse_label(label)

            if parsed is None:
                if current is not None:
                    spans.append(current)
                    current = None
            else:
                if current is None or current.label != parsed:
                    if current is not None:
                        spans.append(current)
                    current = Span(
                        text=text[char_start:char_end],
                        label=parsed,
                        start=char_start,
                        end=char_end,
                    )
                else:
                    current.end = char_end
                    current.text = text[current.start:char_end]

        if current is not None:
            spans.append(current)

        return spans

    @staticmethod
    def _parse_label(label: str) -> str | None:
        """Parse label, stripping BIO prefix. Returns None for 'O'."""
        if label in ("O", "o"):
            return None
        if label.startswith(("B-", "I-")):
            return label[2:]
        return label


class HierarchicalONNXTokenClassifier:
    """
    Hierarchical token classifier using ONNX models.

    Level 1 (primary): classifies tokens into contexts (professional, learning, etc.)
    Level 2 (secondary): classifies tokens into entities per context (work_history, etc.)
    """

    def __init__(self, models_dir: str | Path):
        models_dir = Path(models_dir)

        # Load primary
        primary_dir = models_dir / "primary"
        if not primary_dir.exists():
            raise FileNotFoundError(f"Primary model not found at {primary_dir}")
        self.primary = ONNXTokenClassifier(primary_dir)

        # Load secondary classifiers
        self.secondary: dict[str, ONNXTokenClassifier] = {}
        secondary_dir = models_dir / "secondary"
        if secondary_dir.exists():
            for ctx in SECONDARY_CONTEXTS:
                ctx_dir = secondary_dir / ctx
                if ctx_dir.exists():
                    self.secondary[ctx] = ONNXTokenClassifier(ctx_dir)
                    logger.info(f"Loaded secondary classifier: {ctx}")

        logger.info(
            f"HierarchicalONNXTokenClassifier ready: "
            f"primary + {len(self.secondary)} secondary classifiers"
        )

    def classify(self, message: str) -> HierarchicalResult:
        """
        Run hierarchical classification on a message.

        1. Primary: label each token with a context (professional, learning, etc.)
        2. For each context found, run the secondary classifier on the span text
           to get entity-level labels.
        """
        result = HierarchicalResult()

        # Step 1: Primary classification
        result.primary_spans = self.primary.classify_spans(message)

        # Deduplicate active contexts preserving order
        seen = set()
        for span in result.primary_spans:
            if span.label not in seen:
                result.active_contexts.append(span.label)
                seen.add(span.label)

        # Step 2: Secondary classification per context
        for ctx in result.active_contexts:
            if ctx not in self.secondary:
                continue

            secondary_clf = self.secondary[ctx]
            ctx_entity_spans: list[Span] = []

            # Get all primary spans for this context
            for primary_span in result.primary_spans:
                if primary_span.label != ctx:
                    continue

                # Run secondary on the span text
                sub_spans = secondary_clf.classify_spans(primary_span.text)

                # Remap offsets to original message coordinates
                for sub_span in sub_spans:
                    sub_span.start += primary_span.start
                    sub_span.end += primary_span.start
                    sub_span.text = message[sub_span.start:sub_span.end]
                    ctx_entity_spans.append(sub_span)

            if ctx_entity_spans:
                result.entity_spans[ctx] = ctx_entity_spans

        return result


# Module-level singleton
_classifier_instance: HierarchicalONNXTokenClassifier | None = None


def get_hierarchical_classifier(models_dir: str | Path | None = None) -> HierarchicalONNXTokenClassifier:
    """Get or create the singleton HierarchicalONNXTokenClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        if models_dir is None:
            raise ValueError("models_dir required on first call")
        _classifier_instance = HierarchicalONNXTokenClassifier(models_dir)
    return _classifier_instance


def set_hierarchical_classifier(clf: HierarchicalONNXTokenClassifier):
    """Set the singleton classifier (used by main.py preload)."""
    global _classifier_instance
    _classifier_instance = clf
