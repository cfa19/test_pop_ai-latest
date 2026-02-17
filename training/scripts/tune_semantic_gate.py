"""
Tune Hierarchical Semantic Gate Thresholds

This script finds optimal similarity thresholds for both primary and secondary
categories using hierarchical classifiers and category-specific centroids.

Algorithm:
1. Load CSV data with message, category, subcategory (including off_topic category)
2. Load hierarchical classifiers (primary + secondary) to predict categories for off-topic
3. Compute centroids for both primary categories and subcategories
4. Tune thresholds hierarchically:
   - Primary level: Tune thresholds for main categories vs off-topic
   - Secondary level: Tune thresholds for subcategories within each category
5. Result: Hierarchical thresholds for two-stage semantic filtering

Goal: Maximize off-topic rejection while minimizing false positives at both levels.

CSV Format Expected:
    message,category,subcategory
    "I want to be a manager",aspirational,dream_roles
    "What is Python?",off_topic,
    "I'm good at SQL",professional,skills

Usage:
    python training/scripts/tune_semantic_gate.py \
        --data-dir training/data/processed \
        --hierarchical-model training/models/hierarchical/20260201_203858 \
        --output training/results/semantic_gate_hierarchical_tuning.json \
        --model all-MiniLM-L6-v2 \
        --min-domain-acceptance 0.95
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install: pip install sentence-transformers scikit-learn pandas")
    sys.exit(1)

# Try to import hierarchical classifier components
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: transformers/torch not available (hierarchical classification disabled)")
    TRANSFORMERS_AVAILABLE = False

# Categories that skip secondary classification
SKIP_SECONDARY_CATEGORIES = {"rag_query", "chitchat", "off_topic"}


def load_messages_from_file(file_path: str) -> List[str]:
    """Load messages from a text file (one per line)."""
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        messages = [line.strip() for line in f if line.strip()]

    return messages


def load_hierarchical_data(data_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]], List[str]]:
    """
    Load hierarchical data from CSV files, including off-topic messages.

    Supports two CSV formats:
    - Hierarchical: message,contexts,entities,sub_entities (pipe-separated)
      Files: all_contexts.csv, professional.csv, learning.csv, etc.
    - Flat: message,category_type,subcategory
      Files: rag_query.csv, chitchat.csv, off_topic.csv

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Tuple of:
        - primary_messages: Dict mapping routing category to list of messages
          (professional, learning, social, psychological, personal, rag_query, chitchat)
        - secondary_messages: Dict mapping context -> entity -> list of messages
        - offtopic_messages: List of off-topic messages
    """
    print(f"Loading hierarchical data from: {data_dir}")

    primary_messages = defaultdict(list)
    secondary_messages = defaultdict(lambda: defaultdict(list))
    offtopic_messages = []

    # 1. Load hierarchical context CSVs (message,contexts,entities,sub_entities)
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue

        if 'contexts' in df.columns and 'entities' in df.columns:
            # Hierarchical CSV: message,contexts,entities,sub_entities
            for _, row in df.iterrows():
                msg = str(row['message'])
                contexts = str(row['contexts']).split('|') if pd.notna(row['contexts']) else []
                entities = str(row['entities']).split('|') if pd.notna(row['entities']) else []

                primary_ctx = contexts[0] if contexts else None
                if primary_ctx:
                    primary_messages[primary_ctx].append(msg)

                    # Map entities to their context for secondary
                    for entity in entities:
                        secondary_messages[primary_ctx][entity].append(msg)

            print(f"  Loaded {len(df)} hierarchical messages from {filename}")

        elif 'category_type' in df.columns:
            # Flat CSV: message,category_type,subcategory
            cat_type = df['category_type'].iloc[0] if len(df) > 0 else filename.replace('.csv', '')

            if cat_type == 'off_topic':
                offtopic_messages.extend(df['message'].tolist())
                print(f"  Loaded {len(df)} off-topic messages from {filename}")
            else:
                for _, row in df.iterrows():
                    primary_messages[cat_type].append(str(row['message']))
                print(f"  Loaded {len(df)} {cat_type} messages from {filename}")

        else:
            print(f"  Warning: {filename} has unknown format (columns: {list(df.columns)}), skipping")

    # Convert defaultdicts to regular dicts
    primary_messages = dict(primary_messages)
    secondary_messages = {ctx: dict(ents) for ctx, ents in secondary_messages.items()}

    # Summary
    total_domain = sum(len(msgs) for msgs in primary_messages.values())
    print(f"\nTotal messages loaded:")
    print(f"  Off-topic: {len(offtopic_messages)}")
    print(f"  Domain: {total_domain}")
    for category, msgs in sorted(primary_messages.items()):
        print(f"    {category}: {len(msgs)} messages")
        if category in secondary_messages:
            for entity, entity_msgs in sorted(secondary_messages[category].items()):
                print(f"      → {entity}: {len(entity_msgs)} messages")

    return primary_messages, secondary_messages, offtopic_messages


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


def load_hierarchical_classifier(model_base_path: str):
    """
    Load hierarchical classifiers (routing + entity classifiers per context).

    New model structure:
        routing/final/                    - 8-class routing
        contexts/final/                   - 5-class context (not used here)
        <context>/entities/final/         - entity classifier per context

    Args:
        model_base_path: Base path to hierarchical model directory

    Returns:
        Dict with 'routing' and 'entities' classifiers, or None
    """
    if not TRANSFORMERS_AVAILABLE:
        print("  WARNING: transformers not available, cannot load hierarchical classifier")
        return None

    print(f"\n  Loading hierarchical classifiers from: {model_base_path}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load routing classifier (routing/final/)
        routing_path = os.path.join(model_base_path, "routing", "final")
        print(f"    Loading routing classifier from {routing_path}...")
        routing_tokenizer = AutoTokenizer.from_pretrained(routing_path, local_files_only=True)
        routing_model = AutoModelForSequenceClassification.from_pretrained(routing_path, local_files_only=True)
        routing_model.to(device)
        routing_model.eval()

        with open(os.path.join(routing_path, "label_mappings.json"), "r") as f:
            routing_labels = json.load(f)

        print(f"    ✓ Routing classifier loaded ({len(routing_labels)} categories)")

        # Load entity classifiers per context (<context>/entities/final/)
        entity_classifiers = {}
        context_names = ["professional", "learning", "social", "psychological", "personal"]

        for ctx in context_names:
            entities_path = os.path.join(model_base_path, ctx, "entities", "final")
            if not os.path.exists(entities_path):
                continue

            try:
                print(f"    Loading entity classifier for {ctx}...")
                tokenizer = AutoTokenizer.from_pretrained(entities_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(entities_path, local_files_only=True)
                model.to(device)
                model.eval()

                with open(os.path.join(entities_path, "label_mappings.json"), "r") as f:
                    labels = json.load(f)

                entity_classifiers[ctx] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'labels': labels,
                    'device': device
                }
                print(f"      ✓ Loaded ({len(labels)} entities)")

            except Exception as e:
                print(f"      ✗ Failed to load {ctx} entities classifier: {e}")

        print(f"    ✓ Loaded {len(entity_classifiers)} entity classifiers")

        return {
            'primary': {
                'model': routing_model,
                'tokenizer': routing_tokenizer,
                'labels': routing_labels,
                'device': device
            },
            'secondary': entity_classifiers
        }

    except Exception as e:
        print(f"  ERROR loading hierarchical classifier: {e}")
        return None


def _resolve_label(labels: dict, pred_id: int) -> str:
    """Resolve a predicted ID to a label name, handling multiple label_mappings formats."""
    # Format 1: {"id2label": {"0": "professional", ...}}
    if "id2label" in labels:
        return labels["id2label"][str(pred_id)]
    # Format 2: {"label_mappings": {"0": "professional", ...}, "threshold": ...}
    if "label_mappings" in labels:
        return labels["label_mappings"][str(pred_id)]
    # Format 3: {"0": "professional", ...} (direct mapping)
    return labels[str(pred_id)]


def classify_with_hierarchical(message: str, classifiers: Dict) -> Tuple[str, str | None]:
    """
    Classify message using hierarchical classifier.

    Args:
        message: Message to classify
        classifiers: Dictionary with primary and secondary classifiers

    Returns:
        Tuple of (primary_category, subcategory)
    """
    if not classifiers:
        return "unknown", None

    # Primary classification
    primary = classifiers['primary']
    inputs = primary['tokenizer'](message, return_tensors="pt", truncation=True, max_length=512, padding=True).to(primary['device'])

    with torch.no_grad():
        outputs = primary['model'](**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        primary_category = _resolve_label(primary['labels'], pred_id)

    # Secondary classification (if available)
    subcategory = None
    if primary_category in classifiers['secondary']:
        secondary = classifiers['secondary'][primary_category]
        inputs = secondary['tokenizer'](message, return_tensors="pt", truncation=True, max_length=512, padding=True).to(secondary['device'])

        with torch.no_grad():
            outputs = secondary['model'](**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            subcategory = _resolve_label(secondary['labels'], pred_id)

    return primary_category, subcategory


def classify_offtopic_hierarchical(offtopic_messages: List[str], classifiers: Dict) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    Classify off-topic messages using hierarchical classifier.

    Args:
        offtopic_messages: List of off-topic messages
        classifiers: Hierarchical classifiers

    Returns:
        Tuple of:
        - primary_grouped: Dict mapping primary category to messages
        - secondary_grouped: Dict mapping primary -> subcategory -> messages
    """
    if not classifiers:
        print("  WARNING: No classifiers available, skipping classification")
        return {"unknown": offtopic_messages}, {}

    print(f"  Classifying {len(offtopic_messages)} off-topic messages (hierarchical)...")

    primary_grouped = defaultdict(list)
    secondary_grouped = defaultdict(lambda: defaultdict(list))

    for i, message in enumerate(offtopic_messages):
        if (i + 1) % 100 == 0:
            print(f"    Classified {i + 1}/{len(offtopic_messages)} messages...")

        try:
            primary_cat, subcat = classify_with_hierarchical(message, classifiers)
            primary_grouped[primary_cat].append(message)

            if subcat:
                secondary_grouped[primary_cat][subcat].append(message)

        except Exception as e:
            print(f"    Warning: Classification failed: {e}")
            primary_grouped["unknown"].append(message)

    print("\n  ✓ Off-topic messages grouped by predicted category:")
    for category, messages in sorted(primary_grouped.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"    {category}: {len(messages)} messages")
        if category in secondary_grouped:
            for subcat, subcat_msgs in secondary_grouped[category].items():
                print(f"      -> {subcat}: {len(subcat_msgs)} messages")

    return dict(primary_grouped), {k: dict(v) for k, v in secondary_grouped.items()}


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


def tune_hierarchical_thresholds(
    primary_messages: Dict[str, List[str]],
    secondary_messages: Dict[str, Dict[str, List[str]]],
    offtopic_primary_grouped: Dict[str, List[str]],
    offtopic_secondary_grouped: Dict[str, Dict[str, List[str]]],
    model: SentenceTransformer,
    min_domain_acceptance: float = 0.95
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    """
    Tune thresholds hierarchically for both primary and secondary categories.

    Args:
        primary_messages: Dict mapping primary category to messages
        secondary_messages: Dict mapping primary -> subcategory -> messages
        offtopic_primary_grouped: Off-topic messages grouped by predicted primary category
        offtopic_secondary_grouped: Off-topic messages grouped by primary -> subcategory
        model: SentenceTransformer model
        min_domain_acceptance: Minimum domain acceptance rate

    Returns:
        Tuple of (primary_results, secondary_results, primary_centroids, secondary_centroids)
    """
    print("\n" + "=" * 80)
    print("HIERARCHICAL THRESHOLD TUNING")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Tune Primary Category Thresholds
    # =========================================================================

    print("\n[PRIMARY LEVEL] Tuning primary category thresholds...")
    print("-" * 80)

    # Compute primary centroids
    primary_centroids = {}
    for category, messages in primary_messages.items():
        embeddings = compute_embeddings(messages, model)
        centroid = compute_centroid(embeddings)
        primary_centroids[category] = centroid
        print(f"  ✓ {category}: centroid from {len(messages)} messages")

    # Tune threshold for each primary category
    primary_results = {}

    for category in primary_messages.keys():
        print(f"\n[PRIMARY: {category.upper()}]")

        category_domain_messages = primary_messages[category]
        category_offtopic_messages = offtopic_primary_grouped.get(category, [])

        if not category_offtopic_messages:
            # Use sample from all off-topic
            all_offtopic = []
            for msgs in offtopic_primary_grouped.values():
                all_offtopic.extend(msgs)
            category_offtopic_messages = all_offtopic[:min(500, len(all_offtopic))]

        print(f"  Domain: {len(category_domain_messages)} messages")
        print(f"  Off-topic: {len(category_offtopic_messages)} messages")

        # Compute similarities
        domain_embeddings = compute_embeddings(category_domain_messages, model)
        offtopic_embeddings = compute_embeddings(category_offtopic_messages, model)

        category_centroid = primary_centroids[category]
        domain_similarities = cosine_similarity(domain_embeddings, category_centroid).flatten()
        offtopic_similarities = cosine_similarity(offtopic_embeddings, category_centroid).flatten()

        # Find optimal threshold
        optimal_threshold, optimal_metrics = find_optimal_threshold(
            domain_similarities, offtopic_similarities, min_domain_acceptance=min_domain_acceptance
        )

        print(f"  ✓ Threshold: {optimal_threshold:.4f}")
        print(f"    Domain acceptance: {optimal_metrics['domain_acceptance_rate'] * 100:.2f}%")
        print(f"    Off-topic rejection: {optimal_metrics['offtopic_rejection_rate'] * 100:.2f}%")

        primary_results[category] = {
            "threshold": optimal_threshold,
            "metrics": optimal_metrics,
            "domain_similarity_stats": {
                "mean": float(np.mean(domain_similarities)),
                "std": float(np.std(domain_similarities)),
            },
        }

    # =========================================================================
    # STEP 2: Tune Secondary Category Thresholds
    # =========================================================================

    print("\n" + "=" * 80)
    print("[SECONDARY LEVEL] Tuning subcategory thresholds...")
    print("-" * 80)

    secondary_results = {}
    secondary_centroids = {}

    for primary_category, subcategories_dict in secondary_messages.items():
        if primary_category in SKIP_SECONDARY_CATEGORIES:
            continue

        print(f"\n[SECONDARY: {primary_category.upper()}]")
        secondary_results[primary_category] = {}
        secondary_centroids[primary_category] = {}

        # Compute secondary centroids for this primary category
        for subcategory, messages in subcategories_dict.items():
            print(f"\n  [{primary_category} -> {subcategory}]")

            # Get off-topic messages for this subcategory
            subcat_offtopic = []
            if primary_category in offtopic_secondary_grouped:
                subcat_offtopic = offtopic_secondary_grouped[primary_category].get(subcategory, [])

            if not subcat_offtopic:
                # Use all off-topic from this primary category
                if primary_category in offtopic_primary_grouped:
                    subcat_offtopic = offtopic_primary_grouped[primary_category][:min(200, len(offtopic_primary_grouped[primary_category]))]

            print(f"    Domain: {len(messages)} messages")
            print(f"    Off-topic: {len(subcat_offtopic)} messages")

            if len(subcat_offtopic) < 10:
                print("    ⚠️ Too few off-topic messages, skipping")
                continue

            # Compute embeddings
            domain_embeddings = compute_embeddings(messages, model)
            offtopic_embeddings = compute_embeddings(subcat_offtopic, model)

            # Compute subcategory centroid
            subcat_centroid = compute_centroid(domain_embeddings)

            # Store centroid
            secondary_centroids[primary_category][subcategory] = subcat_centroid

            # Compute similarities
            domain_similarities = cosine_similarity(domain_embeddings, subcat_centroid).flatten()
            offtopic_similarities = cosine_similarity(offtopic_embeddings, subcat_centroid).flatten()

            # Find optimal threshold
            optimal_threshold, optimal_metrics = find_optimal_threshold(
                domain_similarities, offtopic_similarities, min_domain_acceptance=min_domain_acceptance
            )

            print(f"    ✓ Threshold: {optimal_threshold:.4f}")
            print(f"      Domain acceptance: {optimal_metrics['domain_acceptance_rate'] * 100:.2f}%")
            print(f"      Off-topic rejection: {optimal_metrics['offtopic_rejection_rate'] * 100:.2f}%")

            secondary_results[primary_category][subcategory] = {
                "threshold": optimal_threshold,
                "metrics": optimal_metrics,
                "domain_similarity_stats": {
                    "mean": float(np.mean(domain_similarities)),
                    "std": float(np.std(domain_similarities)),
                },
            }

    return primary_results, secondary_results, primary_centroids, secondary_centroids


def save_hierarchical_results(
    output_path: str,
    primary_results: Dict[str, Dict],
    secondary_results: Dict[str, Dict[str, Dict]],
    primary_centroids: Dict[str, np.ndarray],
    secondary_centroids: Dict[str, Dict[str, np.ndarray]],
    offtopic_primary_grouped: Dict[str, List[str]],
    model_path: str,
    model_name: str
):
    """Save hierarchical tuning results to JSON file and centroids to pickle files."""

    # Compute global metrics
    primary_metrics = {
        "mean_threshold": np.mean([r["threshold"] for r in primary_results.values()]),
        "mean_domain_acceptance": np.mean([r["metrics"]["domain_acceptance_rate"] for r in primary_results.values()]),
        "mean_offtopic_rejection": np.mean([r["metrics"]["offtopic_rejection_rate"] for r in primary_results.values()]),
        "mean_f1_score": np.mean([r["metrics"]["f1_score"] for r in primary_results.values()]),
    }

    # Count total secondary categories
    total_secondary = sum(len(subcats) for subcats in secondary_results.values())

    results = {
        "model_name": model_name,
        "model_path": model_path,
        "approach": "hierarchical_thresholds",
        "hierarchical": True,
        "primary_categories": len(primary_results),
        "secondary_categories": total_secondary,
        "global_metrics": {
            "primary": primary_metrics,
            "mean_domain_acceptance": primary_metrics["mean_domain_acceptance"],
            "mean_offtopic_rejection": primary_metrics["mean_offtopic_rejection"],
        },
        "primary_thresholds": {
            category: result["threshold"] for category, result in primary_results.items()
        },
        "secondary_thresholds": {
            category: {subcat: result["threshold"] for subcat, result in subcats.items()}
            for category, subcats in secondary_results.items()
        },
        "detailed_results": {
            "primary": primary_results,
            "secondary": secondary_results,
        },
        "offtopic_distribution": {
            "primary": {category: len(messages) for category, messages in offtopic_primary_grouped.items()},
        },
    }

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save thresholds and metrics to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Thresholds saved to: {output_path}")

    # Save centroids to models directory
    centroids_dir = Path(model_path) / "semantic_gate"
    centroids_dir.mkdir(parents=True, exist_ok=True)

    # Save primary centroids
    primary_centroids_path = centroids_dir / "primary_centroids.pkl"
    with open(primary_centroids_path, "wb") as f:
        pickle.dump(primary_centroids, f)
    print(f"✓ Primary centroids saved to: {primary_centroids_path}")

    # Save secondary centroids
    secondary_centroids_path = centroids_dir / "secondary_centroids.pkl"
    with open(secondary_centroids_path, "wb") as f:
        pickle.dump(secondary_centroids, f)
    print(f"✓ Secondary centroids saved to: {secondary_centroids_path}")




def main():
    parser = argparse.ArgumentParser(description="Tune hierarchical semantic gate thresholds for both primary and secondary categories")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing CSV files with message,category,subcategory (includes off_topic)")
    parser.add_argument(
        "--hierarchical-model", type=str, default=None, help="Path to hierarchical model directory (optional, for classifying off-topic)"
    )
    parser.add_argument("--output", type=str, default="training/results/semantic_gate_hierarchical_tuning.json", help="Output path for results")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--min-domain-acceptance", type=float, default=0.95, help="Minimum domain acceptance rate (default: 0.95)")

    args = parser.parse_args()

    print("=" * 80)
    print("HIERARCHICAL SEMANTIC GATE THRESHOLD TUNING")
    print("=" * 80)

    # 1. Load data
    print("\n[1/5] Loading hierarchical data...")
    primary_messages, secondary_messages, offtopic_messages = load_hierarchical_data(args.data_dir)

    if not offtopic_messages:
        print(f"ERROR: No off-topic messages found in {args.data_dir}")
        print("  Make sure CSV files contain messages with category='off_topic'")
        sys.exit(1)

    if not primary_messages:
        print(f"ERROR: No domain messages found in {args.data_dir}")
        sys.exit(1)

    print("\nData Summary:")
    print(f"  Off-topic messages: {len(offtopic_messages):,}")
    print(f"  Primary categories: {len(primary_messages)}")
    total_domain = sum(len(msgs) for msgs in primary_messages.values())
    print(f"  Total domain messages: {total_domain:,}")
    total_secondary = sum(len(subcats) for subcats in secondary_messages.values())
    print(f"  Secondary categories: {total_secondary}")

    # 2. Load embedding model
    print(f"\n[2/5] Loading embedding model: {args.model}...")
    model = SentenceTransformer(args.model)
    print(f"  ✓ Model loaded (embedding dim: {model.get_sentence_embedding_dimension()})")

    # 3. Classify off-topic messages using hierarchical classifier (if provided)
    print("\n[3/5] Classifying off-topic messages...")
    if args.hierarchical_model and TRANSFORMERS_AVAILABLE:
        classifiers = load_hierarchical_classifier(args.hierarchical_model)
        if classifiers:
            offtopic_primary_grouped, offtopic_secondary_grouped = classify_offtopic_hierarchical(offtopic_messages, classifiers)
        else:
            print("  Classifier loading failed, using ungrouped off-topic messages")
            offtopic_primary_grouped = {cat: offtopic_messages for cat in primary_messages.keys()}
            offtopic_secondary_grouped = {}
    else:
        if args.hierarchical_model and not TRANSFORMERS_AVAILABLE:
            print("  WARNING: Model path provided but transformers not available")
        print("  Using all off-topic messages for each category (no classification)")
        offtopic_primary_grouped = {cat: offtopic_messages for cat in primary_messages.keys()}
        offtopic_secondary_grouped = {}

    # 4. Tune hierarchical thresholds
    print("\n[4/5] Tuning hierarchical thresholds...")
    primary_results, secondary_results, primary_centroids, secondary_centroids = tune_hierarchical_thresholds(
        primary_messages,
        secondary_messages,
        offtopic_primary_grouped,
        offtopic_secondary_grouped,
        model,
        min_domain_acceptance=args.min_domain_acceptance
    )

    # 5. Save results
    print("\n[5/5] Saving results...")
    save_hierarchical_results(
        args.output,
        primary_results,
        secondary_results,
        primary_centroids,
        secondary_centroids,
        offtopic_primary_grouped,
        args.hierarchical_model or "",
        args.model
    )

    # 6. Print summary
    print_hierarchical_summary(primary_results, secondary_results, offtopic_primary_grouped)

    print("\n✅ Hierarchical tuning complete!")


def print_hierarchical_summary(
    primary_results: Dict[str, Dict],
    secondary_results: Dict[str, Dict[str, Dict]],
    offtopic_primary_grouped: Dict[str, List[str]]
):
    """Print summary of hierarchical tuning results."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL SEMANTIC GATE TUNING RESULTS")
    print("=" * 80)

    # Primary thresholds
    print("\n📊 PRIMARY CATEGORY THRESHOLDS:")
    print(f"\n{'Category':<20} {'Threshold':<12} {'Domain Acc':<12} {'Off-topic Rej':<15} {'F1 Score':<10}")
    print("-" * 80)

    for category, result in sorted(primary_results.items()):
        threshold = result["threshold"]
        metrics = result["metrics"]
        print(
            f"{category:<20} {threshold:<12.4f} {metrics['domain_acceptance_rate'] * 100:<11.2f}% "
            f"{metrics['offtopic_rejection_rate'] * 100:<14.2f}% {metrics['f1_score']:<10.4f}"
        )

    # Compute primary averages
    mean_threshold = np.mean([r["threshold"] for r in primary_results.values()])
    mean_acceptance = np.mean([r["metrics"]["domain_acceptance_rate"] for r in primary_results.values()])
    mean_rejection = np.mean([r["metrics"]["offtopic_rejection_rate"] for r in primary_results.values()])
    mean_f1 = np.mean([r["metrics"]["f1_score"] for r in primary_results.values()])

    print("-" * 80)
    print(f"{'AVERAGE':<20} {mean_threshold:<12.4f} {mean_acceptance * 100:<11.2f}% {mean_rejection * 100:<14.2f}% {mean_f1:<10.4f}")

    # Secondary thresholds
    if secondary_results:
        print("\n📊 SECONDARY CATEGORY THRESHOLDS:")

        for category in sorted(secondary_results.keys()):
            print(f"\n  [{category.upper()}]")
            print(f"  {'Subcategory':<25} {'Threshold':<12} {'Domain Acc':<12} {'Off-topic Rej':<15}")
            print("  " + "-" * 75)

            for subcat, result in sorted(secondary_results[category].items()):
                threshold = result["threshold"]
                metrics = result["metrics"]
                print(
                    f"  {subcat:<25} {threshold:<12.4f} {metrics['domain_acceptance_rate'] * 100:<11.2f}% "
                    f"{metrics['offtopic_rejection_rate'] * 100:<14.2f}%"
                )

    # Off-topic distribution
    print("\n🔍 OFF-TOPIC MESSAGE DISTRIBUTION:")
    total_offtopic = sum(len(msgs) for msgs in offtopic_primary_grouped.values())
    for category, messages in sorted(offtopic_primary_grouped.items(), key=lambda x: len(x[1]), reverse=True):
        percentage = len(messages) / total_offtopic * 100
        print(f"  {category:<20} {len(messages):>5} messages ({percentage:>5.1f}%)")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("  Primary thresholds:")
    print("  {")
    for category, result in sorted(primary_results.items()):
        print(f"    '{category}': {result['threshold']:.4f},")
    print("  }")

    if secondary_results:
        print("\n  Secondary thresholds:")
        print("  {")
        for category, subcats in sorted(secondary_results.items()):
            print(f"    '{category}': {{")
            for subcat, result in sorted(subcats.items()):
                print(f"      '{subcat}': {result['threshold']:.4f},")
            print("    },")
        print("  }")

    # Additional insights
    print("\n📝 INSIGHTS:")

    # Find categories with lowest acceptance
    lowest_acceptance_category = min(primary_results.items(), key=lambda x: x[1]["metrics"]["domain_acceptance_rate"])
    print(
        f"  Lowest domain acceptance: {lowest_acceptance_category[0]} "
        f"({lowest_acceptance_category[1]['metrics']['domain_acceptance_rate'] * 100:.2f}%)"
    )
    print(f"    Threshold: {lowest_acceptance_category[1]['threshold']:.4f}")

    # Find categories with lowest rejection
    lowest_rejection_category = min(primary_results.items(), key=lambda x: x[1]["metrics"]["offtopic_rejection_rate"])
    print(
        f"\n  Lowest off-topic rejection: {lowest_rejection_category[0]} "
        f"({lowest_rejection_category[1]['metrics']['offtopic_rejection_rate'] * 100:.2f}%)"
    )
    print("    → Off-topic messages most commonly misclassified as this category")

    # Check separation
    print("\n  Per-category domain vs off-topic separation:")
    for category, result in sorted(primary_results.items()):
        domain_mean = result["domain_similarity_stats"]["mean"]
        if "offtopic_similarity_stats" in result:
            offtopic_mean = result.get("offtopic_similarity_stats", {}).get("mean", 0)
            separation = domain_mean - offtopic_mean
            status = "✓" if separation > 0.2 else "⚠️" if separation > 0.1 else "❌"
            print(f"    {status} {category:<20} {separation:>6.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
