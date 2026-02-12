"""
Test Hierarchical Intent Classifier

This script demonstrates how to use the trained hierarchical classifiers:
1. Primary classifier predicts the main category
2. Secondary classifier predicts the subcategory within that category
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HierarchicalClassifier:
    """
    Hierarchical intent classifier with primary and secondary levels.
    """

    def __init__(self, model_dir: str):
        """
        Initialize hierarchical classifier.

        Args:
            model_dir: Directory containing primary and secondary models
                       (should have 'primary/final' and 'secondary/<category>/final' subdirectories)
        """
        self.model_dir = model_dir

        # Load primary classifier
        primary_path = os.path.join(model_dir, "primary", "final")
        print(f"Loading primary classifier from {primary_path}...")

        self.primary_tokenizer = AutoTokenizer.from_pretrained(primary_path)
        self.primary_model = AutoModelForSequenceClassification.from_pretrained(primary_path)
        self.primary_model.eval()

        # Load primary label mappings
        with open(os.path.join(primary_path, "label_mappings.json"), "r") as f:
            primary_mappings = json.load(f)
            self.primary_id2label = {int(k): v for k, v in primary_mappings["id2label"].items()}
            self.primary_categories = primary_mappings["categories"]

        print(f"  Primary categories: {len(self.primary_categories)}")

        # Load secondary classifiers
        self.secondary_tokenizers = {}
        self.secondary_models = {}
        self.secondary_id2label = {}

        secondary_dir = os.path.join(model_dir, "secondary")
        if os.path.exists(secondary_dir):
            for category in os.listdir(secondary_dir):
                category_path = os.path.join(secondary_dir, category, "final")
                if not os.path.exists(category_path):
                    continue

                print(f"Loading secondary classifier for '{category}'...")
                self.secondary_tokenizers[category] = AutoTokenizer.from_pretrained(category_path)
                self.secondary_models[category] = AutoModelForSequenceClassification.from_pretrained(category_path)
                self.secondary_models[category].eval()

                # Load label mappings
                with open(os.path.join(category_path, "label_mappings.json"), "r") as f:
                    mappings = json.load(f)
                    self.secondary_id2label[category] = {int(k): v for k, v in mappings["id2label"].items()}

        print(f"  Secondary classifiers loaded: {len(self.secondary_models)}")
        print(f"    Categories: {', '.join(sorted(self.secondary_models.keys()))}")

    def predict_primary(self, message: str) -> tuple:
        """
        Predict primary category.

        Args:
            message: Input message

        Returns:
            Tuple of (category, confidence)
        """
        inputs = self.primary_tokenizer(
            message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.primary_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        category = self.primary_id2label[pred_idx]
        return category, confidence

    def predict_secondary(self, message: str, category: str) -> tuple:
        """
        Predict subcategory within a given category.

        Args:
            message: Input message
            category: Primary category

        Returns:
            Tuple of (subcategory, confidence) or (None, None) if no secondary classifier
        """
        if category not in self.secondary_models:
            return None, None

        tokenizer = self.secondary_tokenizers[category]
        model = self.secondary_models[category]
        id2label = self.secondary_id2label[category]

        inputs = tokenizer(
            message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()

        subcategory = id2label[pred_idx]
        return subcategory, confidence

    def predict(self, message: str) -> dict:
        """
        Predict both primary category and subcategory.

        Args:
            message: Input message

        Returns:
            Dictionary with category, subcategory, and confidence scores
        """
        # Predict primary
        category, primary_conf = self.predict_primary(message)

        # Predict secondary (if available)
        subcategory, secondary_conf = self.predict_secondary(message, category)

        return {
            "message": message,
            "category": category,
            "category_confidence": primary_conf,
            "subcategory": subcategory,
            "subcategory_confidence": secondary_conf
        }

    def predict_batch(self, messages: list) -> list:
        """
        Predict categories for a batch of messages.

        Args:
            messages: List of input messages

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(msg) for msg in messages]


def main():
    parser = argparse.ArgumentParser(description="Test hierarchical intent classifier")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained hierarchical models"
    )
    parser.add_argument(
        "--message",
        type=str,
        help="Single message to classify"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing messages (one per line)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (enter messages one at a time)"
    )

    args = parser.parse_args()

    # Load classifier
    classifier = HierarchicalClassifier(args.model_dir)

    print(f"\n{'='*60}")
    print("HIERARCHICAL CLASSIFIER READY")
    print(f"{'='*60}\n")

    if args.message:
        # Single message
        result = classifier.predict(args.message)
        print(f"Message: {result['message']}")
        print(f"Category: {result['category']} (confidence: {result['category_confidence']:.3f})")
        if result['subcategory']:
            print(f"Subcategory: {result['subcategory']} (confidence: {result['subcategory_confidence']:.3f})")
        else:
            print(f"Subcategory: None (no secondary classifier for {result['category']})")

    elif args.file:
        # File with messages
        with open(args.file, "r", encoding="utf-8") as f:
            messages = [line.strip() for line in f if line.strip()]

        print(f"Classifying {len(messages)} messages from {args.file}...\n")
        results = classifier.predict_batch(messages)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['message']}")
            print(f"   Category: {result['category']} ({result['category_confidence']:.3f})")
            if result['subcategory']:
                print(f"   Subcategory: {result['subcategory']} ({result['subcategory_confidence']:.3f})")
            print()

    elif args.interactive:
        # Interactive mode
        print("Interactive mode - enter messages to classify (Ctrl+C to exit)")
        print()

        try:
            while True:
                message = input("Enter message: ").strip()
                if not message:
                    continue

                result = classifier.predict(message)
                print(f"  Category: {result['category']} (confidence: {result['category_confidence']:.3f})")
                if result['subcategory']:
                    print(f"  Subcategory: {result['subcategory']} (confidence: {result['subcategory_confidence']:.3f})")
                else:
                    print("  Subcategory: None")
                print()

        except KeyboardInterrupt:
            print("\nExiting...")

    else:
        print("Error: Specify --message, --file, or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()
