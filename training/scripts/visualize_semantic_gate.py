"""
Visualize Semantic Gate Tuning Results

Creates plots to visualize semantic gate performance across different thresholds.

Usage:
    python training/scripts/visualize_semantic_gate.py \
        --results training/results/semantic_gate_tuning.json \
        --output training/results/semantic_gate_plots.png
"""

import argparse
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install: pip install matplotlib numpy")
    sys.exit(1)


def load_results(results_path: str) -> dict:
    """Load tuning results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def is_per_category_format(results: dict) -> bool:
    """Check if results use per-category thresholds (from tune_semantic_gate.py)."""
    return results.get("use_per_category_thresholds", False) and "per_category_results" in results


def plot_metrics_vs_threshold(results: dict, ax):
    """Plot key metrics vs threshold."""
    thresholds = [r["threshold"] for r in results["threshold_sweep"]]
    domain_acceptance = [r["domain_acceptance_rate"] for r in results["threshold_sweep"]]
    offtopic_rejection = [r["offtopic_rejection_rate"] for r in results["threshold_sweep"]]
    f1_scores = [r["f1_score"] for r in results["threshold_sweep"]]

    ax.plot(thresholds, domain_acceptance, label="Domain Acceptance", linewidth=2, color="#2ecc71")
    ax.plot(thresholds, offtopic_rejection, label="Off-topic Rejection", linewidth=2, color="#e74c3c")
    ax.plot(thresholds, f1_scores, label="F1 Score", linewidth=2, color="#3498db", linestyle="--")

    # Mark optimal threshold
    optimal_threshold = results["optimal_threshold"]
    ax.axvline(x=optimal_threshold, color="black", linestyle=":", linewidth=2, label=f"Optimal ({optimal_threshold:.3f})")

    ax.set_xlabel("Similarity Threshold", fontsize=12)
    ax.set_ylabel("Rate / Score", fontsize=12)
    ax.set_title("Semantic Gate Performance vs Threshold", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)


def plot_confusion_at_optimal(results: dict, ax):
    """Plot confusion matrix at optimal threshold."""
    metrics = results["optimal_metrics"]

    tp = metrics["true_positives"]
    fn = metrics["false_negatives"]
    tn = metrics["true_negatives"]
    fp = metrics["false_positives"]

    # Confusion matrix
    cm = np.array([[tp, fn], [fp, tn]])

    # Normalize by row (actual class)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_normalized[i, j] < 0.5 else "black"
            ax.text(
                j, i, f"{cm[i, j]:,}\n({cm_normalized[i, j] * 100:.1f}%)", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold"
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Accepted", "Rejected"])
    ax.set_yticklabels(["Domain", "Off-topic"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix (Threshold={results['optimal_threshold']:.3f})", fontsize=14, fontweight="bold")

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_category_similarities(results: dict, ax):
    """Plot similarity distribution for each category."""
    category_stats = results["category_statistics"]

    categories = list(category_stats.keys())
    means = [category_stats[cat]["mean"] for cat in categories]
    stds = [category_stats[cat]["std"] for cat in categories]
    mins = [category_stats[cat]["min"] for cat in categories]
    maxs = [category_stats[cat]["max"] for cat in categories]

    # Sort by mean similarity
    sorted_indices = np.argsort(means)
    categories = [categories[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    mins = [mins[i] for i in sorted_indices]
    maxs = [maxs[i] for i in sorted_indices]

    y_pos = np.arange(len(categories))

    # Plot mean with error bars (std)
    ax.barh(y_pos, means, xerr=stds, align="center", alpha=0.7, color="#3498db", edgecolor="black")

    # Plot min/max range
    for i, (cat, mean, min_val, max_val) in enumerate(zip(categories, means, mins, maxs)):
        ax.plot([min_val, max_val], [i, i], "k-", linewidth=2, alpha=0.5)
        ax.plot([min_val], [i], "ko", markersize=4)
        ax.plot([max_val], [i], "ko", markersize=4)

    # Mark optimal threshold
    optimal_threshold = results["optimal_threshold"]
    ax.axvline(x=optimal_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({optimal_threshold:.3f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Similarity to Domain Centroid", fontsize=12)
    ax.set_title("Category Similarity Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, 1)


def plot_precision_recall_curve(results: dict, ax):
    """Plot precision-recall curve across thresholds."""
    precisions = [r["precision"] for r in results["threshold_sweep"]]
    recalls = [r["recall"] for r in results["threshold_sweep"]]
    thresholds = [r["threshold"] for r in results["threshold_sweep"]]

    # Plot PR curve
    ax.plot(recalls, precisions, linewidth=2, color="#9b59b6")

    # Mark optimal threshold
    optimal_metrics = results["optimal_metrics"]
    ax.plot(
        optimal_metrics["recall"],
        optimal_metrics["precision"],
        "ro",
        markersize=12,
        label=f"Optimal (F1={optimal_metrics['f1_score']:.3f})",
        zorder=5,
    )

    # Add some threshold annotations
    for i in [10, 25, 40]:  # Sample thresholds
        if i < len(thresholds):
            ax.annotate(f"{thresholds[i]:.2f}", xy=(recalls[i], precisions[i]), xytext=(5, 5), textcoords="offset points", fontsize=8, alpha=0.7)

    ax.set_xlabel("Recall (Domain Acceptance)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)


# --- Per-category format (from tune_semantic_gate.py) ---


def plot_per_category_thresholds(results: dict, ax):
    """Plot per-category thresholds and metrics (bar charts)."""
    pcr = results["per_category_results"]
    categories = list(pcr.keys())
    thresholds = [pcr[c]["threshold"] for c in categories]
    domain_acceptance = [pcr[c]["metrics"]["domain_acceptance_rate"] for c in categories]
    offtopic_rejection = [pcr[c]["metrics"]["offtopic_rejection_rate"] for c in categories]
    f1_scores = [pcr[c]["metrics"]["f1_score"] for c in categories]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, thresholds, width, label="Threshold", color="#3498db")
    bars2 = ax.bar(x, domain_acceptance, width, label="Domain Acceptance", color="#2ecc71")
    bars3 = ax.bar(x + width, offtopic_rejection, width, label="Off-topic Rejection", color="#e74c3c")
    bars4 = ax.bar(x + 2 * width, f1_scores, width, label="F1 Score", color="#9b59b6")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Rate / Score", fontsize=12)
    ax.set_title("Per-Category Thresholds and Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)


def plot_aggregated_confusion_per_category(results: dict, ax):
    """Plot aggregated confusion matrix (sum across all categories)."""
    pcr = results["per_category_results"]
    tp = sum(pcr[c]["metrics"]["true_positives"] for c in pcr)
    fn = sum(pcr[c]["metrics"]["false_negatives"] for c in pcr)
    tn = sum(pcr[c]["metrics"]["true_negatives"] for c in pcr)
    fp = sum(pcr[c]["metrics"]["false_positives"] for c in pcr)

    cm = np.array([[tp, fn], [fp, tn]])
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_normalized[i, j] < 0.5 else "black"
            ax.text(
                j, i, f"{cm[i, j]:,}\n({cm_normalized[i, j] * 100:.1f}%)", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold"
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Accepted", "Rejected"])
    ax.set_yticklabels(["Domain", "Off-topic"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    mean_thresh = results["global_metrics"]["mean_threshold"]
    ax.set_title(f"Aggregated Confusion Matrix (Mean Threshold={mean_thresh:.3f})", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_per_category_similarities(results: dict, ax):
    """Plot similarity distribution per category with per-category threshold lines."""
    pcr = results["per_category_results"]
    categories = list(pcr.keys())
    means = [pcr[c]["domain_similarity_stats"]["mean"] for c in categories]
    stds = [pcr[c]["domain_similarity_stats"]["std"] for c in categories]
    mins = [pcr[c]["domain_similarity_stats"]["min"] for c in categories]
    maxs = [pcr[c]["domain_similarity_stats"]["max"] for c in categories]
    thresholds = [pcr[c]["threshold"] for c in categories]

    sorted_indices = np.argsort(means)
    categories = [categories[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    mins = [mins[i] for i in sorted_indices]
    maxs = [maxs[i] for i in sorted_indices]
    thresholds = [thresholds[i] for i in sorted_indices]

    y_pos = np.arange(len(categories))
    ax.barh(y_pos, means, xerr=stds, align="center", alpha=0.7, color="#3498db", edgecolor="black")

    for i, (cat, mean, min_val, max_val, thresh) in enumerate(zip(categories, means, mins, maxs, thresholds)):
        ax.plot([min_val, max_val], [i, i], "k-", linewidth=2, alpha=0.5)
        ax.plot([min_val], [i], "ko", markersize=4)
        ax.plot([max_val], [i], "ko", markersize=4)
        ax.plot([thresh], [i], "r|", markersize=14, markeredgewidth=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Similarity to Category Centroid", fontsize=12)
    ax.set_title("Per-Category Similarity Distribution (red = threshold)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, 1)


def plot_per_category_precision_recall(results: dict, ax):
    """Plot (recall, precision) point per category."""
    pcr = results["per_category_results"]
    categories = list(pcr.keys())
    recalls = [pcr[c]["metrics"]["recall"] for c in categories]
    precisions = [pcr[c]["metrics"]["precision"] for c in categories]
    f1_scores = [pcr[c]["metrics"]["f1_score"] for c in categories]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    for i, (cat, r, p, f1) in enumerate(zip(categories, recalls, precisions, f1_scores)):
        ax.scatter(r, p, s=100, c=[colors[i]], label=f"{cat} (F1={f1:.2f})", edgecolors="black", zorder=5)

    ax.set_xlabel("Recall (Domain Acceptance)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall by Category", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)


def create_comprehensive_plot(results: dict, output_path: str):
    """Create comprehensive visualization with multiple subplots."""
    per_category = is_per_category_format(results)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    if per_category:
        # 1. Per-category thresholds and metrics (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        plot_per_category_thresholds(results, ax1)

        # 2. Aggregated confusion matrix (middle-left)
        ax2 = fig.add_subplot(gs[1, 0])
        plot_aggregated_confusion_per_category(results, ax2)

        # 3. Precision-Recall by category (middle-right)
        ax3 = fig.add_subplot(gs[1, 1])
        plot_per_category_precision_recall(results, ax3)

        # 4. Per-category similarity distribution (bottom, full width)
        ax4 = fig.add_subplot(gs[2, :])
        plot_per_category_similarities(results, ax4)

        main_plot_fn = plot_per_category_thresholds
    else:
        # Legacy single-threshold format
        ax1 = fig.add_subplot(gs[0, :])
        plot_metrics_vs_threshold(results, ax1)
        ax2 = fig.add_subplot(gs[1, 0])
        plot_confusion_at_optimal(results, ax2)
        ax3 = fig.add_subplot(gs[1, 1])
        plot_precision_recall_curve(results, ax3)
        ax4 = fig.add_subplot(gs[2, :])
        plot_category_similarities(results, ax4)
        main_plot_fn = plot_metrics_vs_threshold

    # Add overall title
    fig.suptitle(
        f"Semantic Gate Tuning Results - Model: {results['model_name']}" + (" (per-category)" if per_category else ""),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")

    # Also save individual plots
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    fig1, ax = plt.subplots(figsize=(10, 6))
    main_plot_fn(results, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    print(f"Individual plots saved to: {output_dir}/")


def print_summary_table(results: dict):
    """Print a summary table of key metrics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if is_per_category_format(results):
        pcr = results["per_category_results"]
        gm = results["global_metrics"]

        print("\nPER-CATEGORY THRESHOLDS:")
        print(f"\n{'Category':<20} {'Threshold':<12} {'Domain Acc':<12} {'Off-topic Rej':<14} {'F1':<10}")
        print("-" * 70)
        for category, r in sorted(pcr.items()):
            m = r["metrics"]
            print(
                f"{category:<20} {r['threshold']:<12.4f} {m['domain_acceptance_rate'] * 100:<11.2f}% "
                f"{m['offtopic_rejection_rate'] * 100:<13.2f}% {m['f1_score']:<10.4f}"
            )
        print("-" * 70)
        print(
            f"{'MEAN':<20} {gm['mean_threshold']:<12.4f} {gm['mean_domain_acceptance'] * 100:<11.2f}% "
            f"{gm['mean_offtopic_rejection'] * 100:<13.2f}% {gm['mean_f1_score']:<10.4f}"
        )

        # Aggregated counts
        tp = sum(r["metrics"]["true_positives"] for r in pcr.values())
        fn = sum(r["metrics"]["false_negatives"] for r in pcr.values())
        tn = sum(r["metrics"]["true_negatives"] for r in pcr.values())
        fp = sum(r["metrics"]["false_positives"] for r in pcr.values())
        total = tp + fn + tn + fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / total if total > 0 else 0

        print("\nAGGREGATED METRICS:")
        print(f"  Domain Acceptance: {rec * 100:.2f}%  |  Off-topic Rejection: {tn / (tn + fp) * 100:.2f}%" if (tn + fp) > 0 else "  (N/A)")
        print(f"  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}  |  Accuracy: {acc:.4f}")
        print(f"  TP: {tp:,}  FN: {fn:,}  TN: {tn:,}  FP: {fp:,}")
    else:
        metrics = results.get("optimal_metrics") or results.get("global_metrics", {})
        print(f"\n{'Metric':<30} {'Value':<20}")
        print("-" * 50)
        opt = results.get("optimal_threshold")
        print(f"{'Optimal Threshold':<30} {opt:.4f}" if opt is not None else f"{'Optimal Threshold':<30} N/A")
        print(f"{'Domain Acceptance Rate':<30} {metrics.get('domain_acceptance_rate', 0) * 100:.2f}%")
        print(f"{'Off-topic Rejection Rate':<30} {metrics.get('offtopic_rejection_rate', 0) * 100:.2f}%")
        print(f"{'Precision':<30} {metrics.get('precision', 0):.4f}")
        print(f"{'Recall':<30} {metrics.get('recall', 0):.4f}")
        print(f"{'F1 Score':<30} {metrics.get('f1_score', 0):.4f}")
        print(f"{'Accuracy':<30} {metrics.get('accuracy', 0):.4f}")
        print()
        print(f"{'True Positives (Domain)':<30} {metrics.get('true_positives', 0):,}")
        print(f"{'False Negatives (Domain)':<30} {metrics.get('false_negatives', 0):,}")
        print(f"{'True Negatives (Off-topic)':<30} {metrics.get('true_negatives', 0):,}")
        print(f"{'False Positives (Off-topic)':<30} {metrics.get('false_positives', 0):,}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize semantic gate tuning results")
    parser.add_argument("--results", type=str, required=True, help="Path to tuning results JSON file")
    parser.add_argument("--output", type=str, default="training/results/semantic_gate_plots.png", help="Output path for visualization")

    args = parser.parse_args()

    print("=" * 80)
    print("SEMANTIC GATE VISUALIZATION")
    print("=" * 80)

    # Load results
    print(f"\nLoading results from: {args.results}")
    results = load_results(args.results)

    # Print summary
    print_summary_table(results)

    # Create visualization
    print("\nCreating visualizations...")
    create_comprehensive_plot(results, args.output)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
