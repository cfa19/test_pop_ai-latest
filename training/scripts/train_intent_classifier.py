"""
Train Intent Classifier Model

Fine-tune a Hugging Face transformer (BERT, DistilBERT, RoBERTa) for
intent classification to replace the LLM-based intent classifier node.
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
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

# Category mapping for BERT classifier
# Includes: RAG queries, 6 Store A contexts, and chitchat
# NOTE: off_topic is NOT included - it's detected by the semantic gate
INTENT_CATEGORIES = ["rag_query", "professional", "psychological", "learning", "social", "emotional", "aspirational", "chitchat"]

# "Real intents" used for centroid computation (excludes chitchat)
# These represent the core domain topics for semantic similarity checks
REAL_INTENT_CATEGORIES = ["rag_query", "professional", "psychological", "learning", "social", "emotional", "aspirational"]

LABEL2ID = {label: idx for idx, label in enumerate(INTENT_CATEGORIES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_data_from_directory(data_dir: str, val_split: float = 0.2) -> DatasetDict:
    """
    Load training data from directory containing category-specific files.

    Supports both .txt files (one message per line) and .csv files (with message and category columns).

    Args:
        data_dir: Directory containing category files (e.g., rag_queries.txt, professional.txt, etc.)
        val_split: Fraction of data to use for validation (default: 0.2)

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print(f"Loading data from directory: {data_dir}")

    # Category files to load (8 categories - excluding off_topic)
    # off_topic is detected by the semantic gate, not trained in the classifier
    # Try both .txt and .csv extensions
    category_names = ["rag_queries", "professional", "psychological", "learning", "social", "emotional", "aspirational", "chitchat"]

    print("NOTE: off_topic data is NOT loaded for classifier training.")
    print("      Off-topic detection is handled by the semantic gate (Stage 1).")

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
        else:
            print(f"  Warning: {category_name}.txt and {category_name}.csv not found, skipping")

    if not all_dfs:
        raise ValueError(f"No category files found in {data_dir}")

    # Combine all dataframes
    train_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal examples loaded: {len(train_df)}")

    # Validate required columns
    if "message" not in train_df.columns or "category" not in train_df.columns:
        raise ValueError("Data must have 'message' and 'category' columns")

    # Map categories to labels
    train_df["label"] = train_df["category"].map(LABEL2ID)

    # Remove unknown categories
    train_df = train_df.dropna(subset=["label"])
    train_df["label"] = train_df["label"].astype(int)

    # Split into train and validation sets
    print(f"Splitting data into train/validation with {val_split * 100:.1f}% validation split...")
    train_split_df, val_split_df = train_test_split(train_df, test_size=val_split, stratify=train_df["label"], random_state=42)

    train_dataset = Dataset.from_pandas(train_split_df[["message", "label"]])
    val_dataset = Dataset.from_pandas(val_split_df[["message", "label"]])
    train_labels = train_split_df["label"]
    val_labels = val_split_df["label"]

    datasets = {"train": train_dataset, "validation": val_dataset}

    print(f"\nTrain examples: {len(datasets['train'])}")
    print(f"Validation examples: {len(datasets['validation'])}")

    # Print label distribution
    print("\nLabel distribution (train):")
    for label_id, label_name in ID2LABEL.items():
        count = (train_labels == label_id).sum()
        print(f"  {label_name}: {count} ({count / len(train_labels) * 100:.1f}%)")

    print("\nLabel distribution (validation):")
    for label_id, label_name in ID2LABEL.items():
        count = (val_labels == label_id).sum()
        print(f"  {label_name}: {count} ({count / len(val_labels) * 100:.1f}%)")

    return DatasetDict(datasets), train_df


def load_data(train_path: str, val_path: str = None, val_split: float = 0.2):
    """
    Load training and validation data from a single CSV file or directory.

    Args:
        train_path: Path to training CSV or directory containing category files
        val_path: Optional path to validation CSV. If None, splits train data.
        val_split: Fraction of data to use for validation (default: 0.2)

    Returns:
        Tuple of (DatasetDict with 'train' and 'validation' splits, full training DataFrame)
    """
    # Check if train_path is a directory
    if os.path.isdir(train_path):
        return load_data_from_directory(train_path, val_split)

    print(f"Loading data from {train_path}...")

    # Load training data
    train_df = pd.read_csv(train_path)

    if "message" not in train_df.columns or "category" not in train_df.columns:
        raise ValueError("CSV must have 'message' and 'category' columns")

    # Map categories to labels
    train_df["label"] = train_df["category"].map(LABEL2ID)

    # Remove unknown categories
    train_df = train_df.dropna(subset=["label"])
    train_df["label"] = train_df["label"].astype(int)

    if val_path:
        # Load separate validation data
        print(f"Loading validation data from {val_path}...")
        val_df = pd.read_csv(val_path)

        if "message" not in val_df.columns or "category" not in val_df.columns:
            raise ValueError("Validation CSV must have 'message' and 'category' columns")

        val_df["label"] = val_df["category"].map(LABEL2ID)
        val_df = val_df.dropna(subset=["label"])
        val_df["label"] = val_df["label"].astype(int)

        train_dataset = Dataset.from_pandas(train_df[["message", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["message", "label"]])
        train_labels = train_df["label"]
        val_labels = val_df["label"]
    else:
        # Split training data into train and validation sets
        print(f"Splitting data into train/validation with {val_split * 100:.1f}% validation split...")
        train_split_df, val_split_df = train_test_split(train_df, test_size=val_split, stratify=train_df["label"], random_state=42)

        train_dataset = Dataset.from_pandas(train_split_df[["message", "label"]])
        val_dataset = Dataset.from_pandas(val_split_df[["message", "label"]])
        train_labels = train_split_df["label"]
        val_labels = val_split_df["label"]

    datasets = {"train": train_dataset, "validation": val_dataset}

    print(f"\nTrain examples: {len(datasets['train'])}")
    print(f"Validation examples: {len(datasets['validation'])}")

    # Print label distribution
    print("\nLabel distribution (train):")
    for label_id, label_name in ID2LABEL.items():
        count = (train_labels == label_id).sum()
        print(f"  {label_name}: {count} ({count / len(train_labels) * 100:.1f}%)")

    print("\nLabel distribution (validation):")
    for label_id, label_name in ID2LABEL.items():
        count = (val_labels == label_id).sum()
        print(f"  {label_name}: {count} ({count / len(val_labels) * 100:.1f}%)")

    return DatasetDict(datasets), train_df


def tokenize_function(examples, tokenizer):
    """Tokenize input messages."""
    return tokenizer(examples["message"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Overall accuracy
    accuracy = accuracy_score(labels, preds)

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    # Classification report - specify all labels to include missing classes
    all_labels = list(range(len(INTENT_CATEGORIES)))
    report = classification_report(labels, preds, labels=all_labels, target_names=INTENT_CATEGORIES, zero_division=0)
    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_centroids(train_df: pd.DataFrame, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, np.ndarray]:
    """
    Compute embedding centroids for real intent categories.

    This is used for the first-stage semantic gate to detect off-topic messages
    by checking similarity against domain-specific centroids.

    Args:
        train_df: Training dataframe with 'message' and 'category' columns
        embedding_model_name: Sentence transformer model to use

    Returns:
        Dictionary mapping category names to centroid embeddings
    """
    print(f"\n{'=' * 60}")
    print("COMPUTING CENTROIDS FOR SEMANTIC GATE")
    print(f"{'=' * 60}")
    print(f"Embedding model: {embedding_model_name}")
    print(f"Computing centroids for {len(REAL_INTENT_CATEGORIES)} real intent categories")

    # Load embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    centroids = {}

    for category in REAL_INTENT_CATEGORIES:
        # Get messages for this category
        category_messages = train_df[train_df["category"] == category]["message"].tolist()

        if not category_messages:
            print(f"  Warning: No messages found for category '{category}', skipping")
            continue

        # Compute embeddings
        print(f"  Computing centroid for '{category}' ({len(category_messages)} messages)...")
        embeddings = embedding_model.encode(category_messages, show_progress_bar=False)

        # Compute centroid (mean of all embeddings)
        centroid = np.mean(embeddings, axis=0)
        centroids[category] = centroid

        print(f"    Centroid shape: {centroid.shape}")

    print(f"\nComputed {len(centroids)} centroids successfully")
    return centroids, embedding_model_name


def train_model(
    model_name: str,
    train_data: Dataset,
    val_data: Dataset = None,
    output_dir: str = "models/checkpoints/intent_classifier",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    use_class_weights: bool = True,
    train_df: pd.DataFrame = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Train intent classification model.

    Args:
        model_name: Hugging Face model name (e.g., 'distilbert-base-uncased')
        train_data: Training dataset
        val_data: Validation dataset (optional)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_class_weights: Whether to use class weights for imbalanced data
    """
    # Add timestamp folder to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTraining model: {model_name}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(INTENT_CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Tokenize datasets
    print("\nTokenizing data...")
    tokenized_train = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    tokenized_val = None
    if val_data:
        tokenized_val = val_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Compute class weights if needed
    class_weights = None
    if use_class_weights:
        labels = np.array(train_data["label"])
        unique_labels = np.unique(labels)

        # Compute weights for classes present in training data
        computed_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)

        # Create full weight tensor for all classes (0-6)
        full_weights = np.ones(len(INTENT_CATEGORIES), dtype=np.float32)
        for idx, label_id in enumerate(unique_labels):
            full_weights[label_id] = computed_weights[idx]

        class_weights = torch.tensor(full_weights, dtype=torch.float32)
        print(f"\nClass weights (all {len(INTENT_CATEGORIES)} classes): {class_weights}")
        print(f"  Present classes: {sorted(unique_labels.tolist())}")
        print(f"  Missing classes: {sorted(set(range(len(INTENT_CATEGORIES))) - set(unique_labels))}")

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch" if val_data else "no",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True if val_data else False,
        metric_for_best_model="f1" if val_data else None,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=3,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
    )

    # Create trainer
    callbacks = []
    if val_data:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate on validation set
    if val_data:
        print("\nEvaluating on validation set...")
        metrics = trainer.evaluate()
        print("\nValidation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    # Save final model
    final_output = os.path.join(output_dir, "final")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"\nModel saved to {final_output}")

    # Compute and save centroids for semantic gate
    if train_df is not None:
        centroids, embedding_model_name = compute_centroids(train_df, embedding_model)

        # Save centroids
        centroids_path = os.path.join(final_output, "centroids.pkl")
        with open(centroids_path, "wb") as f:
            pickle.dump(centroids, f)
        print(f"\nCentroids saved to {centroids_path}")

        # Save metadata about the centroid computation
        metadata = {
            "embedding_model": embedding_model_name,
            "real_intent_categories": REAL_INTENT_CATEGORIES,
            "all_categories": INTENT_CATEGORIES,
            "num_centroids": len(centroids),
        }
        metadata_path = os.path.join(final_output, "centroid_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Centroid metadata saved to {metadata_path}")

        print("\n" + "=" * 60)
        print("TWO-STAGE CLASSIFIER READY")
        print("=" * 60)
        print("\nStage 1: Semantic Gate (Gross Filter)")
        print(f"  - {len(centroids)} centroids for real intents")
        print(f"  - Embedding model: {embedding_model_name}")
        print("  - Filters out off-topic messages")
        print("\nStage 2: BERT Classifier (Fine-Grained)")
        print(f"  - {len(INTENT_CATEGORIES)} in-domain categories (7 real intents + chitchat)")
        print(f"  - Model: {model_name}")
        print("  - Does NOT predict off_topic (handled by Stage 1)")
    else:
        print("\nWarning: train_df not provided, centroids not computed")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Hugging Face model name")
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to training CSV or directory containing category-specific .txt/.csv files"
    )
    parser.add_argument("--val-data", type=str, help="Path to validation CSV (if not provided, train data will be split)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data to use for validation when splitting (default: 0.2)")
    parser.add_argument("--output", type=str, default="models/", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for computing centroids (default: all-MiniLM-L6-v2)",
    )

    args = parser.parse_args()

    # Load data
    datasets, train_df = load_data(train_path=args.train_data, val_path=args.val_data if args.val_data else None, val_split=args.val_split)

    # Train model
    train_model(
        model_name=args.model,
        train_data=datasets.get("train"),
        val_data=datasets.get("validation"),
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_class_weights=not args.no_class_weights,
        train_df=train_df,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
