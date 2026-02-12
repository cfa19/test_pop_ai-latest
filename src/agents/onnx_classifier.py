"""
ONNX-based Intent Classifier using local fine-tuned model.

Uses onnxruntime with the fine-tuned SequenceClassification ONNX model.
Classification via logits (softmax) -- identical to PyTorch but without torch.
All files loaded from disk, no HuggingFace downloads.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.agents.langgraph_workflow import IntentClassification, MessageCategory


class ONNXIntentClassifier:
    def __init__(self, model_path: str):
        model_dir = Path(model_path)

        # Find ONNX model file (prefer quantized)
        onnx_file = model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        print(f"[ONNX] Loading model from {onnx_file}...")
        self.session = ort.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        output_names = [o.name for o in self.session.get_outputs()]
        print(f"[ONNX] Model loaded (inputs: {self.input_names}, outputs: {output_names})")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Load label mappings from config.json
        config_file = model_dir / "config.json"
        with open(config_file) as f:
            config = json.load(f)
        self.id2label = config.get("id2label", {})
        self.num_labels = len(self.id2label)
        print(f"[ONNX] {self.num_labels} labels: {list(self.id2label.values())}")

    async def classify(self, message: str) -> IntentClassification:
        """Classify a message using the ONNX model."""
        # Tokenize
        inputs = self.tokenizer(
            message, return_tensors="np", padding=True, truncation=True, max_length=128
        )

        # Build feed dict
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = inputs.get(
                "token_type_ids", np.zeros_like(inputs["input_ids"])
            )

        # Run inference
        outputs = self.session.run(None, feed)
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Best prediction
        best_idx = int(np.argmax(probs))
        best_label = self.id2label.get(str(best_idx), f"label_{best_idx}")
        confidence = float(probs[best_idx])

        # Secondary categories (if second-best > 0.15)
        sorted_indices = np.argsort(probs)[::-1]
        secondary = []
        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1]
            if probs[second_idx] > 0.15:
                second_label = self.id2label.get(str(int(second_idx)), "")
                try:
                    secondary.append(MessageCategory(second_label))
                except ValueError:
                    pass

        # Map to enum
        try:
            category_enum = MessageCategory(best_label)
        except ValueError:
            category_enum = MessageCategory.CHITCHAT

        return IntentClassification(
            category=category_enum,
            confidence=confidence,
            reasoning=f"ONNX classifier: {best_label} ({confidence:.1%})",
            key_entities={},
            secondary_categories=secondary,
        )


# Singleton
_classifier_instance: Optional[ONNXIntentClassifier] = None


def get_onnx_classifier(model_path: str = "") -> ONNXIntentClassifier:
    """Get or create the ONNX classifier singleton."""
    global _classifier_instance
    if _classifier_instance is None:
        if not model_path:
            from src.config import INTENT_CLASSIFIER_MODEL_PATH

            model_path = INTENT_CLASSIFIER_MODEL_PATH
        _classifier_instance = ONNXIntentClassifier(model_path)
    return _classifier_instance
