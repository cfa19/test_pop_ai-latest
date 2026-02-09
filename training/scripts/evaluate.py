"""
Model Evaluation Script

Evaluate fine-tuned models on test datasets and compare with baseline.
"""

import argparse
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from training.inference import IntentClassifierModel, WorthinessClassifierModel


def load_test_data_from_directory(data_dir: str) -> pd.DataFrame:
    """
    Load test data from directory containing category-specific .txt files.

    Args:
        data_dir: Directory containing category files

    Returns:
        DataFrame with 'message' and 'category' columns
    """
    print(f"Loading test data from directory: {data_dir}")

    # All 8 categories (off_topic is detected by semantic gate, not classifier)
    category_names = ["rag_queries", "professional", "psychological", "learning", "social", "emotional", "aspirational", "chitchat"]

    all_dfs = []
    for category_name in category_names:
        # Map filename to category label
        category_label = category_name if category_name != "rag_queries" else "rag_query"

        # Try .txt file first, then .csv
        txt_path = os.path.join(data_dir, f"{category_name}.txt")
        csv_path = os.path.join(data_dir, f"{category_name}.csv")

        if os.path.exists(txt_path):
            # Load .txt file (one message per line)
            with open(txt_path, "r", encoding="utf-8") as f:
                messages = [line.strip() for line in f if line.strip()]

            df = pd.DataFrame({"message": messages, "category": category_label})
            print(f"  Loaded {len(df)} examples from {category_name}.txt")
            all_dfs.append(df)

        elif os.path.exists(csv_path):
            # Load .csv file
            df = pd.read_csv(csv_path)

            # If CSV doesn't have category column, infer it from filename
            if "category" not in df.columns and "message" in df.columns:
                df["category"] = category_label

            print(f"  Loaded {len(df)} examples from {category_name}.csv")
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No test data files found in {data_dir}")

    # Combine all dataframes
    test_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal test examples: {len(test_df)}")
    print("\nCategory distribution:")
    print(test_df["category"].value_counts().sort_index())

    return test_df


def evaluate_intent_classifier(model_path: str, test_data_path: str, output_dir: str = None, n_eval: int = 1000) -> Dict:
    """
    Evaluate two-stage intent classifier on test data.

    Args:
        model_path: Path to fine-tuned model (with centroids)
        test_data_path: Path to test data directory or CSV
        output_dir: Directory to save plots and reports
        similarity_threshold: Threshold for semantic gate (default: 0.3)

    Returns:
        Dict with evaluation metrics
    """
    print("=" * 60)
    print("Evaluating Two-Stage Intent Classifier")
    print("=" * 60)

    print("Loading BERT classifier...")
    model = IntentClassifierModel(model_path)

    # Load test data
    if os.path.isdir(test_data_path):
        test_df = load_test_data_from_directory(test_data_path)
    else:
        test_df = pd.read_csv(test_data_path)
        if "message" not in test_df.columns or "category" not in test_df.columns:
            raise ValueError("Test CSV must have 'message' and 'category' columns")

    messages = test_df["message"].tolist()[:n_eval]
    true_labels = test_df["category"].tolist()[:n_eval]

    # Get predictions
    print(f"\nPredicting on {len(messages)} test examples...")
    start_time = time.time()

    predictions = model.predict_batch(messages)
    pred_labels = [p["category"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]

    end_time = time.time()

    # Calculate overall metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)

    # Per-class metrics
    class_report = classification_report(true_labels, pred_labels, zero_division=0)

    # Confusion matrix
    categories = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=categories)

    # Latency
    avg_latency = (end_time - start_time) / len(messages) * 1000  # ms

    # Print results
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg Confidence: {np.mean(confidences):.4f}")
    print(f"Avg Latency: {avg_latency:.2f}ms per message")

    print("\n" + "=" * 60)
    print("PER-CLASS REPORT")
    print("=" * 60)
    print(class_report)

    # Save confusion matrix plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
        plt.title("Confusion Matrix - Intent Classifier")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")

        # Save detailed report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write("Two-Stage Intent Classifier Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test data: {test_data_path}\n")
            f.write(f"Test examples: {len(messages)}\n")
            f.write("\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"Avg Confidence: {np.mean(confidences):.4f}\n")
            f.write(f"Avg Latency: {avg_latency:.2f}ms\n\n")

            f.write("Per-class Report:\n")
            f.write(class_report)
        print(f"Report saved to {report_path}")

        # Save predictions
        pred_path = os.path.join(output_dir, "predictions.csv")
        pred_df_data = {
            "message": messages,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "confidence": confidences,
            "correct": [t == p for t, p in zip(true_labels, pred_labels)],
        }

        results_df = pd.DataFrame(pred_df_data)
        results_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_confidence": np.mean(confidences),
        "avg_latency_ms": avg_latency,
    }

    return results


def evaluate_worthiness_classifier(model_path: str, test_data_path: str, output_dir: str = None) -> Dict:
    """
    Evaluate worthiness classifier on test data.

    Args:
        model_path: Path to fine-tuned model
        test_data_path: Path to test CSV
        output_dir: Directory to save plots and reports

    Returns:
        Dict with evaluation metrics
    """
    print("=" * 60)
    print("Evaluating Worthiness Classifier")
    print("=" * 60)

    # Load model
    model = WorthinessClassifierModel(model_path)

    # Load test data
    test_df = pd.read_csv(test_data_path)
    if "message" not in test_df.columns or "is_worthy" not in test_df.columns:
        raise ValueError("Test CSV must have 'message' and 'is_worthy' columns")

    messages = test_df["message"].tolist()
    true_labels = test_df["is_worthy"].astype(bool).tolist()

    # Get predictions
    print(f"\nPredicting on {len(messages)} test examples...")
    start_time = time.time()
    predictions = model.predict_batch(messages)
    end_time = time.time()

    pred_labels = [p["is_worthy"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]
    scores = [p["score"] for p in predictions]

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Latency
    avg_latency = (end_time - start_time) / len(messages) * 1000  # ms

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg Confidence: {np.mean(confidences):.4f}")
    print(f"Avg Score: {np.mean(scores):.2f}/100")
    print(f"Avg Latency: {avg_latency:.2f}ms per message")
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")

    # Save plots and reports
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Worthy", "Worthy"], yticklabels=["Not Worthy", "Worthy"])
        plt.title("Confusion Matrix - Worthiness Classifier")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")

        # Save report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write("Worthiness Classifier Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test data: {test_data_path}\n")
            f.write(f"Test examples: {len(messages)}\n\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"Avg Confidence: {np.mean(confidences):.4f}\n")
            f.write(f"Avg Score: {np.mean(scores):.2f}/100\n")
            f.write(f"Avg Latency: {avg_latency:.2f}ms\n")
        print(f"Report saved to {report_path}")

        # Save predictions
        pred_path = os.path.join(output_dir, "predictions.csv")
        results_df = pd.DataFrame(
            {
                "message": messages,
                "true_label": true_labels,
                "pred_label": pred_labels,
                "confidence": confidences,
                "score": scores,
                "correct": [t == p for t, p in zip(true_labels, pred_labels)],
            }
        )
        results_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_confidence": np.mean(confidences),
        "avg_score": np.mean(scores),
        "avg_latency_ms": avg_latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument("--task", type=str, required=True, choices=["intent", "worthiness"], help="Task to evaluate")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data directory or CSV file")
    parser.add_argument("--output", type=str, help="Output directory for plots and reports")

    args = parser.parse_args()

    if args.task == "intent":
        evaluate_intent_classifier(model_path=args.model, test_data_path=args.test_data, output_dir=args.output)
    elif args.task == "worthiness":
        evaluate_worthiness_classifier(args.model, args.test_data, args.output)


if __name__ == "__main__":
    main()
