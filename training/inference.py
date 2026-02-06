"""
Inference Module for Fine-tuned Models

Provides easy-to-use interfaces for loading and using fine-tuned models
to replace LangGraph workflow nodes.

Includes:
- IntentClassifierModel: Simple BERT-based intent classifier
- TwoStageIntentClassifier: Hybrid approach with embedding similarity + BERT
- WorthinessClassifierModel: Message worthiness classifier
"""

from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class IntentClassifierModel:
    """
    Intent classifier using fine-tuned BERT/DistilBERT model.

    Replaces the LLM-based intent classifier node in LangGraph workflow.
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the intent classifier.

        Args:
            model_path: Path to fine-tuned model directory
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Resolve to absolute path; HF rejects Windows paths (backslashes, colons) as repo IDs
        self._model_path = str(Path(model_path).resolve())

        print(f"Loading intent classifier from {self._model_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self._model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded. Categories: {list(self.model.config.id2label.values())}")

    def predict(self, text: str, return_probs: bool = False) -> Dict:
        """
        Predict intent category for a message.

        Args:
            text: Input message
            return_probs: Whether to return probabilities for all classes

        Returns:
            Dict with keys:
                - category: Predicted category name
                - confidence: Confidence score (0-1)
                - probabilities: Dict of category -> probability (if return_probs=True)
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Get predicted class
        pred_class_id = torch.argmax(probs).item()
        pred_category = self.model.config.id2label[pred_class_id]
        confidence = probs[pred_class_id].item()

        result = {"category": pred_category, "confidence": float(confidence)}

        # Add all probabilities if requested
        if return_probs:
            result["probabilities"] = {self.model.config.id2label[i]: float(probs[i].item()) for i in range(len(probs))}

        return result

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict intent categories for multiple messages.

        Args:
            texts: List of input messages

        Returns:
            List of prediction dicts
        """
        # Tokenize all inputs
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Convert to list of results
        results = []
        for i in range(len(texts)):
            pred_class_id = torch.argmax(probs[i]).item()
            pred_category = self.model.config.id2label[pred_class_id]
            confidence = probs[i][pred_class_id].item()

            results.append({"category": pred_category, "confidence": float(confidence)})

        return results


class WorthinessClassifierModel:
    """
    Message worthiness classifier using fine-tuned BERT/DistilBERT model.

    Replaces the heuristic/LLM-based worthiness classifier.
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the worthiness classifier.

        Args:
            model_path: Path to fine-tuned model directory
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path_resolved = str(Path(model_path).resolve())

        print(f"Loading worthiness classifier from {model_path_resolved} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_resolved, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path_resolved, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded.")

    def predict(self, text: str) -> Dict:
        """
        Predict if a message is worthy of long-term storage.

        Args:
            text: Input message

        Returns:
            Dict with keys:
                - is_worthy: Boolean prediction
                - confidence: Confidence score (0-1)
                - score: Worthiness score (0-100)
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Binary classification: [not_worthy, worthy]
        is_worthy = probs[1].item() > 0.5
        confidence = max(probs[0].item(), probs[1].item())
        score = int(probs[1].item() * 100)  # Convert to 0-100 scale

        return {"is_worthy": is_worthy, "confidence": float(confidence), "score": score}

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict worthiness for multiple messages.

        Args:
            texts: List of input messages

        Returns:
            List of prediction dicts
        """
        # Tokenize all inputs
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Convert to list of results
        results = []
        for i in range(len(texts)):
            is_worthy = probs[i][1].item() > 0.5
            confidence = max(probs[i][0].item(), probs[i][1].item())
            score = int(probs[i][1].item() * 100)

            results.append({"is_worthy": is_worthy, "confidence": float(confidence), "score": score})

        return results


# Example usage
if __name__ == "__main__":
    # Test intent classifier
    print("=" * 50)
    print("Testing Intent Classifier")
    print("=" * 50)

    model_path = "models/final/intent_classifier"
    classifier = IntentClassifierModel(model_path)

    test_messages = [
        "What is machine learning?",
        "I have 5 years of Python experience",
        "I want to become a data scientist",
        "I'm feeling overwhelmed with work",
    ]

    for msg in test_messages:
        result = classifier.predict(msg, return_probs=True)
        print(f"\nMessage: {msg}")
        print(f"Category: {result['category']} (confidence: {result['confidence']:.2f})")
        if "probabilities" in result:
            print("Top 3 probabilities:")
            sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
            for cat, prob in sorted_probs:
                print(f"  {cat}: {prob:.3f}")

    # Test batch prediction
    print("\n" + "=" * 50)
    print("Testing Batch Prediction")
    print("=" * 50)

    batch_results = classifier.predict_batch(test_messages)
    for msg, result in zip(test_messages, batch_results):
        print(f"{msg[:40]:40} -> {result['category']:15} ({result['confidence']:.2f})")
