"""
Train Hierarchical Intent Classifier

This script trains a two-level hierarchical classifier:
1. Primary classifier: Classifies messages into main categories (8 categories)
2. Secondary classifiers: For each category, classifies into subcategories

Off-topic data is excluded from training; use it later to tune the Semantic gate.

CSV Format Expected:
message,category,subcategory
"I want to be a manager",aspirational,dream_roles
"hi there",chitchat,greetings
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Primary categories (8 main intent categories; off_topic excluded - used for Semantic gate tuning)
PRIMARY_CATEGORIES = [
    "rag_query",
    "professional",
    "psychological",
    "learning",
    "social",
    "emotional",
    "aspirational",
    "chitchat",
]

# Categories that do not get secondary classifiers (primary category is sufficient)
SKIP_SECONDARY_CATEGORIES = {"rag_query", "chitchat"}

# Map primary categories to their subcategories
# This will be populated dynamically from the data
CATEGORY_SUBCATEGORIES = {}


def load_hierarchical_data(data_dir: str, val_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load training data with category and subcategory information.

    Args:
        data_dir: Directory containing CSV files with message, category, subcategory columns
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_df, val_df, category_subcategories_map)
    """
    print(f"Loading hierarchical data from: {data_dir}")
    print("Expected CSV format: message, category, subcategory\n")

    all_dfs = []

    # Load all CSV files in the directory
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)

        # Validate columns
        if 'message' not in df.columns or 'category' not in df.columns or 'subcategory' not in df.columns:
            print(f"  Warning: {filename} missing required columns (message, category, subcategory), skipping")
            continue

        print(f"  Loaded {len(df)} examples from {filename}")
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No valid CSV files found in {data_dir}")

    # Combine all dataframes
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal examples loaded: {len(full_df)}")

    # Exclude off_topic (reserved for Semantic gate tuning)
    off_topic_count = len(full_df[full_df["category"] == "off_topic"])
    full_df = full_df[full_df["category"] != "off_topic"].reset_index(drop=True)
    if off_topic_count:
        print(f"Excluded {off_topic_count} off_topic examples (for Semantic gate tuning)")
    print(f"Examples for training: {len(full_df)}")

    # Build subcategory mapping
    category_subcategories = defaultdict(set)
    for _, row in full_df.iterrows():
        category_subcategories[row['category']].add(row['subcategory'])

    # Convert sets to sorted lists
    category_subcategories = {cat: sorted(list(subcats)) for cat, subcats in category_subcategories.items()}

    print("\nCategory hierarchy:")
    for category, subcategories in sorted(category_subcategories.items()):
        print(f"  {category}: {len(subcategories)} subcategories")
        for subcat in subcategories:
            count = len(full_df[(full_df['category'] == category) & (full_df['subcategory'] == subcat)])
            print(f"    - {subcat}: {count} examples")

    # Split into train and validation
    print(f"\nSplitting data into train/validation ({val_split*100:.1f}% validation)...")
    if full_df.empty:
        raise ValueError("No training data after excluding off_topic. Ensure CSV files contain non-off_topic categories.")
    train_df, val_df = train_test_split(
        full_df,
        test_size=val_split,
        stratify=full_df['category'],  # Stratify by primary category
        random_state=42
    )

    print(f"Train examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")

    return train_df, val_df, category_subcategories


def create_label_mappings(categories: List[str]) -> Tuple[Dict, Dict]:
    """Create label <-> id mappings for a list of categories."""
    label2id = {label: idx for idx, label in enumerate(categories)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def tokenize_function(examples, tokenizer):
    """Tokenize input messages."""
    return tokenizer(
        examples["message"],
        padding="max_length",
        truncation=True,
        max_length=512
    )


def compute_metrics(pred, id2label):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Overall accuracy
    accuracy = accuracy_score(labels, preds)

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    # Classification report
    all_labels = list(range(len(id2label)))
    target_names = [id2label[i] for i in all_labels]
    report = classification_report(
        labels, preds,
        labels=all_labels,
        target_names=target_names,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_classifier(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_column: str,
    categories: List[str],
    output_dir: str,
    num_epochs: float = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    use_class_weights: bool = True,
    classifier_name: str = "classifier",
    eval_every_epochs: float = 1.0
):
    """
    Train a single classifier (either primary or secondary).

    Args:
        model_name: Hugging Face model name
        train_df: Training dataframe with 'message' and label_column
        val_df: Validation dataframe
        label_column: Column name to use as labels ('category' or 'subcategory')
        categories: List of category names for this classifier
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs (supports fractions, e.g., 0.5 = half dataset)
        batch_size: Training batch size
        learning_rate: Learning rate
        use_class_weights: Whether to use class weights
        classifier_name: Name for this classifier (for logging)
        eval_every_epochs: Evaluate on validation every N epochs (e.g. 0.5 = twice per epoch)

    Returns:
        Trained model path
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {classifier_name.upper()}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Categories: {len(categories)}")
    print(f"Train examples: {len(train_df)}")
    print(f"Val examples: {len(val_df)}")
    print(f"Epochs: {num_epochs}")

    # Create label mappings
    label2id, id2label = create_label_mappings(categories)

    # Map labels to ids
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['label'] = train_df[label_column].map(label2id)
    val_df['label'] = val_df[label_column].map(label2id)

    # Remove any unmapped labels
    train_df = train_df.dropna(subset=['label'])
    val_df = val_df.dropna(subset=['label'])
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['message', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['message', 'label']])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize datasets
    print("\nTokenizing data...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    # Compute class weights if needed
    class_weights = None
    if use_class_weights:
        labels = np.array(train_dataset["label"])
        unique_labels = np.unique(labels)

        computed_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )

        full_weights = np.ones(len(categories), dtype=np.float32)
        for idx, label_id in enumerate(unique_labels):
            full_weights[label_id] = computed_weights[idx]

        class_weights = torch.tensor(full_weights, dtype=torch.float32)
        print(f"\nClass weights: {class_weights}")

    # Custom trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()

            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Compute steps per epoch and total steps
    steps_per_epoch = max(1, (len(tokenized_train) + batch_size - 1) // batch_size)
    max_steps = int(steps_per_epoch * num_epochs)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {max_steps} ({num_epochs} epochs)")

    # Compute eval/save strategy: every N epochs or every epoch
    if eval_every_epochs >= 1.0:
        eval_strategy = "epoch"
        save_strategy = "epoch"
        eval_steps = None
        save_steps = None
    else:
        eval_strategy = "steps"
        save_strategy = "steps"
        eval_steps = max(1, int(steps_per_epoch * eval_every_epochs))
        save_steps = eval_steps
        print(f"Eval/save every {eval_steps} steps ({eval_every_epochs} epochs)")

    # Training arguments
    # Use max_steps instead of num_train_epochs to support fractional epochs
    training_args_kw = dict(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        warmup_steps=min(50, max_steps // 10),  # Adjust warmup for small datasets
        fp16=torch.cuda.is_available(),
    )
    if eval_steps is not None:
        training_args_kw["eval_steps"] = eval_steps
        training_args_kw["save_steps"] = save_steps

    training_args = TrainingArguments(**training_args_kw)

    # Create trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=lambda pred: compute_metrics(pred, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate
    print("\nEvaluating on validation set...")
    metrics = trainer.evaluate()
    print("\nValidation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save final model
    final_output = os.path.join(output_dir, "final")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    # Save label mappings
    mappings = {
        "label2id": label2id,
        "id2label": id2label,
        "categories": categories
    }
    with open(os.path.join(final_output, "label_mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"\n{classifier_name} saved to {final_output}")

    return final_output


def compute_centroids(
    train_df: pd.DataFrame,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, np.ndarray]:
    """
    Compute embedding centroids for primary categories (for semantic gate).

    Args:
        train_df: Training dataframe
        embedding_model_name: Sentence transformer model

    Returns:
        Dictionary mapping category names to centroid embeddings
    """
    print(f"\n{'='*60}")
    print("COMPUTING PRIMARY CENTROIDS FOR SEMANTIC GATE")
    print(f"{'='*60}")
    print(f"Embedding model: {embedding_model_name}")

    # Load embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    centroids = {}

    # Compute centroid for each primary category
    for category in PRIMARY_CATEGORIES:
        category_messages = train_df[train_df["category"] == category]["message"].tolist()

        if not category_messages:
            print(f"  Warning: No messages for '{category}', skipping")
            continue

        print(f"  Computing centroid for '{category}' ({len(category_messages)} messages)...")
        embeddings = embedding_model.encode(category_messages, show_progress_bar=False)
        centroid = np.mean(embeddings, axis=0)
        centroids[category] = centroid
        print(f"    Centroid shape: {centroid.shape}")

    print(f"\nComputed {len(centroids)} primary centroids")
    return centroids, embedding_model_name


def compute_secondary_centroids(
    train_df: pd.DataFrame,
    category_subcategories: Dict[str, List[str]],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute embedding centroids for subcategories (for hierarchical semantic gate).

    Args:
        train_df: Training dataframe with category and subcategory columns
        category_subcategories: Mapping of primary categories to their subcategories
        embedding_model_name: Sentence transformer model

    Returns:
        Nested dictionary mapping category -> subcategory -> centroid embedding
    """
    print(f"\n{'='*60}")
    print("COMPUTING SECONDARY CENTROIDS FOR SEMANTIC GATE")
    print(f"{'='*60}")
    print(f"Embedding model: {embedding_model_name}")

    # Load embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    secondary_centroids = {}

    # Compute centroids for each category's subcategories
    for category, subcategories in category_subcategories.items():
        if category in SKIP_SECONDARY_CATEGORIES:
            print(f"\n[{category}] Skipping (no secondary classification)")
            continue

        print(f"\n[{category}] Computing centroids for {len(subcategories)} subcategories...")
        secondary_centroids[category] = {}

        for subcategory in subcategories:
            # Get all messages for this subcategory
            mask = (train_df["category"] == category) & (train_df["subcategory"] == subcategory)
            subcat_messages = train_df[mask]["message"].tolist()

            if not subcat_messages:
                print(f"  Warning: No messages for '{category}/{subcategory}', skipping")
                continue

            print(f"  Computing centroid for '{subcategory}' ({len(subcat_messages)} messages)...")
            embeddings = embedding_model.encode(subcat_messages, show_progress_bar=False)
            centroid = np.mean(embeddings, axis=0)
            secondary_centroids[category][subcategory] = centroid

        total_subcats = len(secondary_centroids[category])
        print(f"  ✓ Computed {total_subcats} centroids for {category}")

    total_secondary = sum(len(subcats) for subcats in secondary_centroids.values())
    print(f"\nComputed {total_secondary} secondary centroids across {len(secondary_centroids)} categories")

    return secondary_centroids, embedding_model_name


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical intent classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing CSV files with message, category, subcategory columns"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model for primary classifier (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--secondary-model",
        type=str,
        default=None,
        help="Hugging Face model for secondary classifiers (default: same as --model)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/models/hierarchical",
        help="Output directory for all trained models"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=5,
        help=(
            "Number of epochs for each classifier (default: 5). "
            "Supports fractions (e.g., 0.5 = half dataset). "
            "Can be overridden by --primary-epochs and --secondary-epochs"
        )
    )
    parser.add_argument(
        "--primary-epochs",
        type=float,
        default=None,
        help="Number of epochs for primary classifier (default: uses --epochs). Supports fractions (e.g., 0.5 = half dataset)"
    )
    parser.add_argument(
        "--secondary-epochs",
        type=float,
        default=None,
        help="Number of epochs for secondary classifiers (default: uses --epochs). Supports fractions (e.g., 0.5 = half dataset)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size (default: 20)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer for centroids (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Skip training secondary classifiers (only train primary)"
    )
    parser.add_argument(
        "--eval-every",
        type=float,
        default=1.0,
        metavar="N",
        help=(
            "Evaluate on validation data every N epochs "
            "(e.g. 0.5 = twice per epoch, default: 1.0). "
            "Can be overridden by --primary-eval-every and --secondary-eval-every"
        )
    )
    parser.add_argument(
        "--primary-eval-every",
        type=float,
        default=None,
        metavar="N",
        help="Evaluate primary classifier every N epochs (default: uses --eval-every)"
    )
    parser.add_argument(
        "--secondary-eval-every",
        type=float,
        default=None,
        metavar="N",
        help="Evaluate secondary classifiers every N epochs (default: uses --eval-every)"
    )
    parser.add_argument(
        "--secondary-only",
        action="store_true",
        help="Skip primary classifier; train only one secondary classifier (requires --category)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="When --secondary-only: category to train (e.g. professional, aspirational). Required with --secondary-only."
    )
    parser.add_argument(
        "--skip-centroids",
        action="store_true",
        help="Skip computing centroids for semantic gate"
    )

    args = parser.parse_args()

    # Set epoch values (use specific if provided, otherwise use general)
    primary_epochs = args.primary_epochs if args.primary_epochs is not None else args.epochs
    secondary_epochs = args.secondary_epochs if args.secondary_epochs is not None else args.epochs

    # Set eval-every values (use specific if provided, otherwise use general)
    primary_eval_every = args.primary_eval_every if args.primary_eval_every is not None else args.eval_every
    secondary_eval_every = args.secondary_eval_every if args.secondary_eval_every is not None else args.eval_every

    # Validate epoch values
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.primary_epochs is not None and args.primary_epochs <= 0:
        parser.error("--primary-epochs must be positive")
    if args.secondary_epochs is not None and args.secondary_epochs <= 0:
        parser.error("--secondary-epochs must be positive")

    # Validate eval-every values
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")
    if args.primary_eval_every is not None and args.primary_eval_every <= 0:
        parser.error("--primary-eval-every must be positive")
    if args.secondary_eval_every is not None and args.secondary_eval_every <= 0:
        parser.error("--secondary-eval-every must be positive")

    if args.secondary_only and not args.category:
        parser.error("--secondary-only requires --category (e.g. --category professional)")

    if args.secondary_only and args.category in SKIP_SECONDARY_CATEGORIES:
        parser.error(
            f"Category '{args.category}' does not have secondary classifiers. "
            f"Skipped categories: {', '.join(sorted(SKIP_SECONDARY_CATEGORIES))}. "
            f"Choose from: professional, psychological, learning, social, emotional, aspirational"
        )

    secondary_model = args.secondary_model or args.model

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("HIERARCHICAL INTENT CLASSIFIER TRAINING")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}\n")

    # Load data
    train_df, val_df, category_subcategories = load_hierarchical_data(
        args.data_dir,
        args.val_split
    )

    # Save category hierarchy metadata
    metadata = {
        "primary_categories": PRIMARY_CATEGORIES,
        "category_subcategories": category_subcategories,
        "train_examples": len(train_df),
        "val_examples": len(val_df),
        "timestamp": timestamp,
        "model": args.model,
        "secondary_model": secondary_model,
        "embedding_model": args.embedding_model,
        "training_config": {
            "primary_epochs": primary_epochs,
            "secondary_epochs": secondary_epochs,
            "primary_eval_every": primary_eval_every,
            "secondary_eval_every": secondary_eval_every,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "use_class_weights": not args.no_class_weights
        }
    }
    with open(os.path.join(output_dir, "hierarchy_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # =========================================================================
    # STEP 1: Train Primary Classifier (skip if --secondary-only)
    # =========================================================================

    primary_model_path = None
    centroids = {}
    centroids_path = None

    if not args.secondary_only:
        primary_output = os.path.join(output_dir, "primary")
        os.makedirs(primary_output, exist_ok=True)

        primary_model_path = train_classifier(
        model_name=args.model,
        train_df=train_df,
        val_df=val_df,
        label_column="category",
        categories=PRIMARY_CATEGORIES,
        output_dir=primary_output,
        num_epochs=primary_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_class_weights=not args.no_class_weights,
        classifier_name="Primary Classifier (8 Categories)",
        eval_every_epochs=primary_eval_every
    )

        # Compute centroids for semantic gate
        if not args.skip_centroids:
            centroids, embedding_model_name = compute_centroids(train_df, args.embedding_model)

            # Save to semantic_gate directory (alongside secondary centroids)
            semantic_gate_dir = os.path.join(output_dir, "semantic_gate")
            os.makedirs(semantic_gate_dir, exist_ok=True)

            centroids_path = os.path.join(semantic_gate_dir, "primary_centroids.pkl")
            with open(centroids_path, "wb") as f:
                pickle.dump(centroids, f)
            print(f"\n✓ Primary centroids saved to: {centroids_path}")

            centroid_metadata = {
                "embedding_model": embedding_model_name,
                "categories": list(centroids.keys()),
                "num_centroids": len(centroids)
            }
            with open(os.path.join(semantic_gate_dir, "primary_centroid_metadata.json"), "w") as f:
                json.dump(centroid_metadata, f, indent=2)
        else:
            centroids_path = None
            centroids = {}
            print("\n--skip-centroids flag set, skipping primary centroid computation")

    if args.skip_secondary and not args.secondary_only:
        print("\n--skip-secondary flag set, skipping secondary classifiers")
        print(f"\nTraining complete! Models saved to: {output_dir}")
        return

    # =========================================================================
    # STEP 2: Train Secondary Classifiers (or single one if --secondary-only)
    # =========================================================================

    print(f"\n{'='*60}")
    print("TRAINING SECONDARY CLASSIFIERS")
    print(f"{'='*60}\n")

    secondary_output = os.path.join(output_dir, "secondary")
    os.makedirs(secondary_output, exist_ok=True)

    secondary_models = {}

    # When --secondary-only, iterate only over the specified category
    categories_to_train = [args.category] if args.secondary_only else sorted(category_subcategories.keys())

    for category in categories_to_train:
        if category not in category_subcategories:
            if args.secondary_only:
                raise ValueError(
                    f"Category '{args.category}' not found in data. "
                    f"Available categories: {', '.join(sorted(category_subcategories.keys()))}"
                )
            continue

        subcategories = category_subcategories[category]
        if category in SKIP_SECONDARY_CATEGORIES:
            print(f"\nSkipping {category}: no secondary classifier (primary category is sufficient)")
            continue
        if len(subcategories) <= 1:
            print(f"\nSkipping {category}: only {len(subcategories)} subcategory (no classification needed)")
            continue

        # Filter data for this category
        category_train_df = train_df[train_df['category'] == category].copy()
        category_val_df = val_df[val_df['category'] == category].copy()

        if len(category_train_df) < 10:
            print(f"\nSkipping {category}: insufficient training data ({len(category_train_df)} examples)")
            continue

        # Train secondary classifier for this category
        category_output = os.path.join(secondary_output, category)
        os.makedirs(category_output, exist_ok=True)

        secondary_model_path = train_classifier(
            model_name=secondary_model,
            train_df=category_train_df,
            val_df=category_val_df,
            label_column="subcategory",
            categories=subcategories,
            output_dir=category_output,
            num_epochs=secondary_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_class_weights=not args.no_class_weights,
            classifier_name=f"Secondary Classifier: {category} ({len(subcategories)} subcategories)",
            eval_every_epochs=secondary_eval_every
        )

        secondary_models[category] = secondary_model_path

    # Save secondary model paths
    secondary_metadata = {
        "secondary_models": secondary_models,
        "secondary_model": secondary_model,
        "categories_trained": list(secondary_models.keys())
    }
    with open(os.path.join(secondary_output, "secondary_metadata.json"), "w") as f:
        json.dump(secondary_metadata, f, indent=2)

    # =========================================================================
    # STEP 3: Compute Secondary Centroids for Semantic Gate
    # =========================================================================

    if not args.skip_centroids and secondary_models:
        print(f"\n{'='*60}")
        print("COMPUTING SECONDARY CENTROIDS")
        print(f"{'='*60}\n")

        secondary_centroids, embedding_model_name = compute_secondary_centroids(
            train_df,
            category_subcategories,
            args.embedding_model
        )

        # Save secondary centroids to semantic_gate directory
        semantic_gate_dir = os.path.join(output_dir, "semantic_gate")
        os.makedirs(semantic_gate_dir, exist_ok=True)

        secondary_centroids_path = os.path.join(semantic_gate_dir, "secondary_centroids.pkl")
        with open(secondary_centroids_path, "wb") as f:
            pickle.dump(secondary_centroids, f)

        print(f"\n✓ Secondary centroids saved to: {secondary_centroids_path}")

        # Save metadata
        secondary_centroid_metadata = {
            "embedding_model": embedding_model_name,
            "num_categories": len(secondary_centroids),
            "total_centroids": sum(len(subcats) for subcats in secondary_centroids.values()),
            "categories": {
                category: len(subcats) for category, subcats in secondary_centroids.items()
            }
        }
        with open(os.path.join(semantic_gate_dir, "secondary_centroid_metadata.json"), "w") as f:
            json.dump(secondary_centroid_metadata, f, indent=2)

        print("✓ Secondary centroid metadata saved")
    else:
        secondary_centroids_path = None
        if args.skip_centroids:
            print("\n--skip-centroids flag set, skipping secondary centroid computation")
        elif not secondary_models:
            print("\nNo secondary models trained, skipping secondary centroid computation")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nAll models saved to: {output_dir}")

    if primary_model_path:
        print(f"\nPrimary classifier: {primary_model_path}")
        print(f"  - {len(PRIMARY_CATEGORIES)} categories: {', '.join(PRIMARY_CATEGORIES)}")

    print(f"\nSecondary classifiers trained: {len(secondary_models)}")
    for category in sorted(secondary_models.keys()):
        num_subcats = len(category_subcategories[category])
        print(f"  - {category}: {num_subcats} subcategories")

    if centroids_path:
        print(f"\nPrimary centroids for semantic gate: {centroids_path}")
        print(f"  - {len(centroids)} primary centroids computed")

    if 'secondary_centroids_path' in locals() and secondary_centroids_path:
        print(f"\nSecondary centroids for semantic gate: {secondary_centroids_path}")
        total_secondary = sum(len(subcats) for subcats in secondary_centroids.values())
        print(f"  - {total_secondary} secondary centroids across {len(secondary_centroids)} categories")

    if not args.secondary_only:
        print("\nUsage:")
        print("  1. Load primary classifier to get main category")
        print("  2. Load corresponding secondary classifier to get subcategory")
        print("  3. Use centroids for semantic gate (off-topic detection)")


if __name__ == "__main__":
    main()
