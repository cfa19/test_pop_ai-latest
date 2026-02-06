"""
Tune Semantic Gate Threshold (Per-Category)

This script finds optimal similarity thresholds for the semantic gate (Stage 1)
using category-specific centroids and classifier-based off-topic grouping.

Algorithm:
1. Load DistilBERT classifier to predict categories for off-topic messages
2. Group off-topic messages by their predicted (wrong) category
3. For each category, compute separate centroid and tune threshold
4. Result: Per-category thresholds that minimize false positives while maximizing off-topic rejection

Goal: Maximize off-topic rejection while minimizing false positives (wrongly rejected in-domain messages).

Usage:
    python training/scripts/tune_semantic_gate.py \
        --offtopic-data training/data/off_topic.txt \
        --domain-data training/data/ \
        --classifier-model training/models/20260201_203858/final \
        --output training/results/semantic_gate_tuning.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install: pip install sentence-transformers scikit-learn")
    sys.exit(1)

# Try to import DistilBERT classifier
try:
    from training.inference import IntentClassifierModel

    DISTILBERT_AVAILABLE = True
except ImportError:
    print("WARNING: DistilBERT classifier not available (transformers not installed)")
    DISTILBERT_AVAILABLE = False


def load_messages_from_file(file_path: str) -> List[str]:
    """Load messages from a text file (one per line)."""
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        messages = [line.strip() for line in f if line.strip()]

    return messages


def load_domain_messages(data_dir: str) -> Dict[str, List[str]]:
    """
    Load all in-domain messages from data directory.

    Returns:
        Dictionary mapping category to list of messages
    """
    categories = ["rag_queries", "professional", "psychological", "learning", "social", "emotional", "aspirational", "chitchat"]

    domain_messages = {}

    for category in categories:
        file_path = os.path.join(data_dir, f"{category}.txt")
        messages = load_messages_from_file(file_path)

        if messages:
            domain_messages[category] = messages
            print(f"  Loaded {len(messages)} messages from {category}")

    return domain_messages


def compute_embeddings(messages: List[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """
    Compute embeddings for a list of messages.

    Args:
        messages: List of text messages
        model: SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        Numpy array of embeddings (n_messages, embedding_dim)
    """
    print(f"  Computing embeddings for {len(messages)} messages...")
    embeddings = model.encode(messages, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute the centroid (mean) of embeddings."""
    return np.mean(embeddings, axis=0, keepdims=True)


def evaluate_threshold(domain_similarities: np.ndarray, offtopic_similarities: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Evaluate performance at a specific threshold.

    Args:
        domain_similarities: Similarities for in-domain messages (should be > threshold)
        offtopic_similarities: Similarities for off-topic messages (should be < threshold)
        threshold: Similarity threshold to test

    Returns:
        Dictionary with performance metrics
    """
    # True positives: In-domain messages correctly accepted
    tp = np.sum(domain_similarities >= threshold)

    # False negatives: In-domain messages wrongly rejected
    fn = np.sum(domain_similarities < threshold)

    # True negatives: Off-topic messages correctly rejected
    tn = np.sum(offtopic_similarities < threshold)

    # False positives: Off-topic messages wrongly accepted
    fp = np.sum(offtopic_similarities >= threshold)

    # Calculate metrics
    total_domain = len(domain_similarities)
    total_offtopic = len(offtopic_similarities)

    # Domain acceptance rate (should be high)
    domain_acceptance_rate = tp / total_domain if total_domain > 0 else 0

    # Off-topic rejection rate (should be high)
    offtopic_rejection_rate = tn / total_offtopic if total_offtopic > 0 else 0

    # Precision: Of all accepted messages, how many were actually in-domain?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall: Of all in-domain messages, how many were correctly accepted?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "threshold": threshold,
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "domain_acceptance_rate": domain_acceptance_rate,
        "offtopic_rejection_rate": offtopic_rejection_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "total_domain": total_domain,
        "total_offtopic": total_offtopic,
    }


def find_optimal_threshold(
    domain_similarities: np.ndarray, offtopic_similarities: np.ndarray, min_domain_acceptance: float = 0.95, target_offtopic_rejection: float = 0.90
) -> Tuple[float, Dict]:
    """
    Find the optimal threshold that:
    1. Accepts at least min_domain_acceptance of in-domain messages
    2. Maximizes off-topic rejection rate

    Args:
        domain_similarities: Similarities for in-domain messages
        offtopic_similarities: Similarities for off-topic messages
        min_domain_acceptance: Minimum acceptable domain acceptance rate (default: 95%)
        target_offtopic_rejection: Target off-topic rejection rate (default: 90%)

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    print("\nFinding optimal threshold...")
    print(f"  Constraint: Domain acceptance >= {min_domain_acceptance * 100:.1f}%")
    print(f"  Target: Off-topic rejection >= {target_offtopic_rejection * 100:.1f}%")

    # Test thresholds from 0.0 to 1.0
    thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = None
    best_metrics = None
    best_score = -1

    for threshold in thresholds:
        metrics = evaluate_threshold(domain_similarities, offtopic_similarities, threshold)

        # Check if domain acceptance meets minimum requirement
        if metrics["domain_acceptance_rate"] >= min_domain_acceptance:
            # Score: prioritize off-topic rejection, then F1
            score = metrics["offtopic_rejection_rate"] * 0.7 + metrics["f1_score"] * 0.3

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics

    if best_threshold is None:
        # If no threshold meets constraint, find the one with best F1
        print("  WARNING: No threshold meets domain acceptance constraint!")
        print("  Falling back to best F1 score...")

        best_f1 = -1
        for threshold in thresholds:
            metrics = evaluate_threshold(domain_similarities, offtopic_similarities, threshold)
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_threshold = threshold
                best_metrics = metrics

    return best_threshold, best_metrics


def analyze_by_category(domain_messages: Dict[str, List[str]], model: SentenceTransformer, global_centroid: np.ndarray) -> Dict[str, Dict]:
    """
    Analyze similarity distribution for each category.

    Returns:
        Dictionary mapping category to statistics
    """
    print("\nAnalyzing per-category similarity distributions...")

    category_stats = {}

    for category, messages in domain_messages.items():
        embeddings = compute_embeddings(messages, model)
        similarities = cosine_similarity(embeddings, global_centroid).flatten()

        stats = {
            "count": len(messages),
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "percentile_5": float(np.percentile(similarities, 5)),
            "percentile_25": float(np.percentile(similarities, 25)),
            "percentile_50": float(np.percentile(similarities, 50)),
            "percentile_75": float(np.percentile(similarities, 75)),
            "percentile_95": float(np.percentile(similarities, 95)),
        }

        category_stats[category] = stats

        print(f"\n  {category}:")
        print(f"    Mean similarity: {stats['mean']:.4f}")
        print(f"    Std deviation: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    5th percentile: {stats['percentile_5']:.4f}")

    return category_stats


def classify_offtopic_messages(offtopic_messages: List[str], classifier_model_path: str) -> Dict[str, List[str]]:
    """
    Classify off-topic messages using DistilBERT to see which category they're closest to.

    Args:
        offtopic_messages: List of off-topic messages
        classifier_model_path: Path to trained DistilBERT model

    Returns:
        Dictionary mapping predicted category to list of messages
    """
    if not DISTILBERT_AVAILABLE:
        print("  WARNING: DistilBERT not available, skipping classifier-based grouping")
        return {"unknown": offtopic_messages}

    print(f"\n  Loading classifier from: {classifier_model_path}")

    try:
        classifier = IntentClassifierModel(classifier_model_path)
    except Exception as e:
        print(f"  ERROR loading classifier: {e}")
        print("  Falling back to ungrouped off-topic messages")
        return {"unknown": offtopic_messages}

    print(f"  Classifying {len(offtopic_messages)} off-topic messages...")

    # Group messages by predicted category
    grouped = defaultdict(list)

    for i, message in enumerate(offtopic_messages):
        if (i + 1) % 100 == 0:
            print(f"    Classified {i + 1}/{len(offtopic_messages)} messages...")

        try:
            prediction = classifier.predict(message)
            predicted_category = prediction["category"]
            grouped[predicted_category].append(message)
        except Exception as e:
            print(f"    Warning: Failed to classify message: {e}")
            grouped["unknown"].append(message)

    print("\n  ✓ Off-topic messages grouped by predicted category:")
    for category, messages in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"    {category}: {len(messages)} messages ({len(messages) / len(offtopic_messages) * 100:.1f}%)")

    return dict(grouped)


def compute_per_category_centroids(domain_messages: Dict[str, List[str]], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    """
    Compute separate centroids for each category.

    Args:
        domain_messages: Dictionary mapping category to messages
        model: SentenceTransformer model

    Returns:
        Dictionary mapping category to centroid embedding
    """
    print("\nComputing per-category centroids...")

    centroids = {}

    for category, messages in domain_messages.items():
        embeddings = compute_embeddings(messages, model)
        centroid = compute_centroid(embeddings)
        centroids[category] = centroid
        print(f"  ✓ {category}: centroid computed from {len(messages)} messages")

    return centroids


def tune_per_category_thresholds(
    domain_messages: Dict[str, List[str]], offtopic_grouped: Dict[str, List[str]], model: SentenceTransformer, min_domain_acceptance: float = 0.95
) -> Dict[str, Dict]:
    """
    Tune separate thresholds for each category.

    Args:
        domain_messages: Dictionary mapping category to in-domain messages
        offtopic_grouped: Dictionary mapping category to off-topic messages classified as that category
        model: SentenceTransformer model
        min_domain_acceptance: Minimum domain acceptance rate

    Returns:
        Dictionary mapping category to tuning results
    """
    print("\n" + "=" * 80)
    print("TUNING PER-CATEGORY THRESHOLDS")
    print("=" * 80)

    # Compute per-category centroids
    centroids = compute_per_category_centroids(domain_messages, model)

    # Tune threshold for each category
    category_results = {}

    for category in domain_messages.keys():
        print(f"\n[{category.upper()}]")
        print("-" * 80)

        # Get domain messages for this category
        category_domain_messages = domain_messages[category]

        # Get off-topic messages misclassified as this category
        category_offtopic_messages = offtopic_grouped.get(category, [])

        if not category_offtopic_messages:
            print(f"  No off-topic messages classified as {category}, using global off-topic pool")
            # Use a sample of all off-topic messages
            all_offtopic = []
            for msgs in offtopic_grouped.values():
                all_offtopic.extend(msgs)
            category_offtopic_messages = all_offtopic[: min(500, len(all_offtopic))]

        print(f"  Domain messages: {len(category_domain_messages)}")
        print(f"  Off-topic messages (misclassified as {category}): {len(category_offtopic_messages)}")

        # Compute embeddings
        domain_embeddings = compute_embeddings(category_domain_messages, model)
        offtopic_embeddings = compute_embeddings(category_offtopic_messages, model)

        # Compute similarities to category centroid
        category_centroid = centroids[category]
        domain_similarities = cosine_similarity(domain_embeddings, category_centroid).flatten()
        offtopic_similarities = cosine_similarity(offtopic_embeddings, category_centroid).flatten()

        print(f"  Domain similarity - Mean: {np.mean(domain_similarities):.4f}, Std: {np.std(domain_similarities):.4f}")
        print(f"  Off-topic similarity - Mean: {np.mean(offtopic_similarities):.4f}, Std: {np.std(offtopic_similarities):.4f}")

        # Find optimal threshold for this category
        optimal_threshold, optimal_metrics = find_optimal_threshold(
            domain_similarities, offtopic_similarities, min_domain_acceptance=min_domain_acceptance, target_offtopic_rejection=0.90
        )

        print(f"  ✓ Optimal threshold: {optimal_threshold:.4f}")
        print(f"    Domain acceptance: {optimal_metrics['domain_acceptance_rate'] * 100:.2f}%")
        print(f"    Off-topic rejection: {optimal_metrics['offtopic_rejection_rate'] * 100:.2f}%")
        print(f"    F1 Score: {optimal_metrics['f1_score']:.4f}")

        # Store results
        category_results[category] = {
            "threshold": optimal_threshold,
            "metrics": optimal_metrics,
            "domain_similarity_stats": {
                "mean": float(np.mean(domain_similarities)),
                "std": float(np.std(domain_similarities)),
                "min": float(np.min(domain_similarities)),
                "max": float(np.max(domain_similarities)),
                "percentile_5": float(np.percentile(domain_similarities, 5)),
            },
            "offtopic_similarity_stats": {
                "mean": float(np.mean(offtopic_similarities)),
                "std": float(np.std(offtopic_similarities)),
                "min": float(np.min(offtopic_similarities)),
                "max": float(np.max(offtopic_similarities)),
                "percentile_95": float(np.percentile(offtopic_similarities, 95)),
            },
            "offtopic_count": len(category_offtopic_messages),
        }

    return category_results


def save_results(
    output_path: str, category_results: Dict[str, Dict], offtopic_grouped: Dict[str, List[str]], model_name: str, use_per_category: bool = True
):
    """Save tuning results to JSON file."""

    # Compute global metrics (average across categories)
    global_metrics = {
        "mean_threshold": np.mean([r["threshold"] for r in category_results.values()]),
        "mean_domain_acceptance": np.mean([r["metrics"]["domain_acceptance_rate"] for r in category_results.values()]),
        "mean_offtopic_rejection": np.mean([r["metrics"]["offtopic_rejection_rate"] for r in category_results.values()]),
        "mean_f1_score": np.mean([r["metrics"]["f1_score"] for r in category_results.values()]),
    }

    results = {
        "model_name": model_name,
        "use_per_category_thresholds": use_per_category,
        "global_metrics": global_metrics,
        "per_category_results": category_results,
        "offtopic_distribution": {category: len(messages) for category, messages in offtopic_grouped.items()},
        "recommendation": {
            "approach": "per-category thresholds" if use_per_category else "global threshold",
            "thresholds": {category: result["threshold"] for category, result in category_results.items()},
            "usage": "Use category-specific thresholds in semantic gate configuration",
        },
    }

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


def print_summary(category_results: Dict[str, Dict], offtopic_grouped: Dict[str, List[str]]):
    """Print summary of per-category tuning results."""
    print("\n" + "=" * 80)
    print("SEMANTIC GATE TUNING RESULTS (PER-CATEGORY)")
    print("=" * 80)

    # Print per-category thresholds
    print("\n📊 OPTIMAL THRESHOLDS BY CATEGORY:")
    print(f"\n{'Category':<20} {'Threshold':<12} {'Domain Acc':<12} {'Off-topic Rej':<15} {'F1 Score':<10}")
    print("-" * 80)

    for category, result in sorted(category_results.items()):
        threshold = result["threshold"]
        metrics = result["metrics"]
        print(
            f"{category:<20} {threshold:<12.4f} {metrics['domain_acceptance_rate'] * 100:<11.2f}% "
            f"{metrics['offtopic_rejection_rate'] * 100:<14.2f}% {metrics['f1_score']:<10.4f}"
        )

    # Compute global statistics
    mean_threshold = np.mean([r["threshold"] for r in category_results.values()])
    mean_acceptance = np.mean([r["metrics"]["domain_acceptance_rate"] for r in category_results.values()])
    mean_rejection = np.mean([r["metrics"]["offtopic_rejection_rate"] for r in category_results.values()])
    mean_f1 = np.mean([r["metrics"]["f1_score"] for r in category_results.values()])

    print("-" * 80)
    print(f"{'AVERAGE':<20} {mean_threshold:<12.4f} {mean_acceptance * 100:<11.2f}% {mean_rejection * 100:<14.2f}% {mean_f1:<10.4f}")

    # Print off-topic distribution
    print("\n🔍 OFF-TOPIC MESSAGE DISTRIBUTION:")
    total_offtopic = sum(len(msgs) for msgs in offtopic_grouped.values())
    for category, messages in sorted(offtopic_grouped.items(), key=lambda x: len(x[1]), reverse=True):
        percentage = len(messages) / total_offtopic * 100
        print(f"  {category:<20} {len(messages):>5} messages ({percentage:>5.1f}%)")

    print("\n💡 RECOMMENDATION:")
    print("  Use per-category thresholds in semantic gate configuration:")
    print("  ")
    print("  category_thresholds = {")
    for category, result in sorted(category_results.items()):
        print(f"      '{category}': {result['threshold']:.4f},")
    print("  }")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Tune semantic gate thresholds (per-category) for optimal off-topic filtering")
    parser.add_argument("--offtopic-data", type=str, required=True, help="Path to off_topic.txt file")
    parser.add_argument("--domain-data", type=str, required=True, help="Directory containing domain message files (rag_queries.txt, etc.)")
    parser.add_argument(
        "--classifier-model", type=str, default=None, help="Path to trained DistilBERT classifier (optional, for grouping off-topic messages)"
    )
    parser.add_argument("--output", type=str, default="training/results/semantic_gate_tuning.json", help="Output path for tuning results")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--min-domain-acceptance", type=float, default=0.95, help="Minimum domain acceptance rate constraint (default: 0.95)")

    args = parser.parse_args()

    print("=" * 80)
    print("SEMANTIC GATE THRESHOLD TUNING (PER-CATEGORY)")
    print("=" * 80)

    # 1. Load data
    print("\n[1/6] Loading data...")
    offtopic_messages = load_messages_from_file(args.offtopic_data)
    domain_messages = load_domain_messages(args.domain_data)

    if not offtopic_messages:
        print(f"ERROR: No off-topic messages found at {args.offtopic_data}")
        sys.exit(1)

    if not domain_messages:
        print(f"ERROR: No domain messages found in {args.domain_data}")
        sys.exit(1)

    print(f"\n  Off-topic messages: {len(offtopic_messages):,}")
    print(f"  Domain categories: {len(domain_messages)}")
    total_domain = sum(len(msgs) for msgs in domain_messages.values())
    print(f"  Total domain messages: {total_domain:,}")

    # 2. Load embedding model
    print(f"\n[2/6] Loading embedding model: {args.model}...")
    model = SentenceTransformer(args.model)
    print(f"  ✓ Model loaded (embedding dim: {model.get_sentence_embedding_dimension()})")

    # 3. Classify off-topic messages using DistilBERT (if classifier provided)
    print("\n[3/6] Classifying off-topic messages...")
    if args.classifier_model and DISTILBERT_AVAILABLE:
        offtopic_grouped = classify_offtopic_messages(offtopic_messages, args.classifier_model)
    else:
        if args.classifier_model and not DISTILBERT_AVAILABLE:
            print("  WARNING: Classifier model path provided but DistilBERT not available")
        print("  Using all off-topic messages for each category (no grouping)")
        # Distribute off-topic messages equally across categories for testing
        offtopic_grouped = {category: offtopic_messages for category in domain_messages.keys()}

    # 4. Tune per-category thresholds
    print("\n[4/6] Tuning per-category thresholds...")
    category_results = tune_per_category_thresholds(domain_messages, offtopic_grouped, model, min_domain_acceptance=args.min_domain_acceptance)

    # 5. Save results
    print("\n[5/6] Saving results...")
    save_results(args.output, category_results, offtopic_grouped, args.model, use_per_category=True)

    # 6. Print summary
    print("\n[6/6] Generating summary...")
    print_summary(category_results, offtopic_grouped)

    # Additional insights
    print("\n📝 INSIGHTS:")

    # Find categories with lowest acceptance
    lowest_acceptance_category = min(category_results.items(), key=lambda x: x[1]["metrics"]["domain_acceptance_rate"])
    print(
        f"  Lowest domain acceptance: {lowest_acceptance_category[0]} "
        f"({lowest_acceptance_category[1]['metrics']['domain_acceptance_rate'] * 100:.2f}%)"
    )
    print(f"    Threshold: {lowest_acceptance_category[1]['threshold']:.4f}")

    # Find categories with lowest rejection
    lowest_rejection_category = min(category_results.items(), key=lambda x: x[1]["metrics"]["offtopic_rejection_rate"])
    print(
        f"\n  Lowest off-topic rejection: {lowest_rejection_category[0]} "
        f"({lowest_rejection_category[1]['metrics']['offtopic_rejection_rate'] * 100:.2f}%)"
    )
    print("    → Off-topic messages most commonly misclassified as this category")

    # Check separation for each category
    print("\n  Per-category domain vs off-topic separation:")
    for category, result in sorted(category_results.items()):
        domain_mean = result["domain_similarity_stats"]["mean"]
        offtopic_mean = result["offtopic_similarity_stats"]["mean"]
        separation = domain_mean - offtopic_mean
        status = "✓" if separation > 0.2 else "⚠️" if separation > 0.1 else "❌"
        print(f"    {status} {category:<20} {separation:>6.4f}")

    print("\n✅ Tuning complete!")


if __name__ == "__main__":
    main()
