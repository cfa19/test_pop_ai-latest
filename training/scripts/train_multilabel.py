"""
Hierarchical Multi-Label BERT Training Script for vast.ai

Trains 3 levels of classifiers:
  Level 1: Context classifier (5 classes, softmax) - professional, learning, social, psychological, personal
  Level 2: Entity classifier (per-context, softmax) - e.g., professional has 6 entities
  Level 3: Sub-entity classifier (per-entity, sigmoid multi-label) - e.g., professional_aspirations has 5 sub-entities

Also trains:
  Level 0: Routing classifier (8 classes, softmax) - 5 contexts + rag_query + chitchat + off_topic

Usage on vast.ai:
  pip install -r requirements-training.txt
  python training/scripts/train_multilabel.py --data-dir training/data/hierarchical --output-dir models/hierarchical
  python training/scripts/train_multilabel.py --level 1 --data-dir training/data/hierarchical  # only Level 1
  python training/scripts/train_multilabel.py --level 3 --context professional --entity professional_aspirations  # specific L3
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ==============================================================================
# DATASET
# ==============================================================================

class SingleLabelDataset(Dataset):
    """Dataset for softmax single-label classification (Level 0, 1, 2)."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class MultiLabelDataset(Dataset):
    """Dataset for sigmoid multi-label classification (Level 3)."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels  # binary vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_hierarchical_csv(filepath):
    """Load hierarchical CSV: message, contexts, entities, sub_entities."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "message": row["message"],
                "contexts": row["contexts"].split("|") if row["contexts"] else [],
                "entities": row["entities"].split("|") if row["entities"] else [],
                "sub_entities": row["sub_entities"].split("|") if row["sub_entities"] else [],
            })
    return rows


def load_flat_csv(filepath):
    """Load flat CSV: message, category_type, subcategory."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "message": row["message"],
                "category_type": row["category_type"],
                "subcategory": row["subcategory"],
            })
    return rows


def prepare_level0_data(data_dir):
    """
    Level 0: Routing classifier (8 classes).
    Classes: professional, learning, social, psychological, personal, rag_query, chitchat, off_topic
    """
    texts, labels = [], []
    label_map = {}
    label_idx = 0

    # Load hierarchical data (5 contexts)
    all_contexts_path = os.path.join(data_dir, "all_contexts.csv")
    if os.path.exists(all_contexts_path):
        rows = load_hierarchical_csv(all_contexts_path)
        for row in rows:
            primary_context = row["contexts"][0] if row["contexts"] else None
            if primary_context:
                if primary_context not in label_map:
                    label_map[primary_context] = label_idx
                    label_idx += 1
                texts.append(row["message"])
                labels.append(label_map[primary_context])

    # Load non-context data
    for nct in ["rag_query", "chitchat", "off_topic"]:
        nct_path = os.path.join(data_dir, f"{nct}.csv")
        if os.path.exists(nct_path):
            rows = load_flat_csv(nct_path)
            if nct not in label_map:
                label_map[nct] = label_idx
                label_idx += 1
            for row in rows:
                texts.append(row["message"])
                labels.append(label_map[nct])

    return texts, labels, label_map


def prepare_level1_data(data_dir):
    """
    Level 1: Context classifier (5 classes, softmax).
    Only uses hierarchical context data.
    """
    texts, labels = [], []
    label_map = {}
    label_idx = 0

    all_contexts_path = os.path.join(data_dir, "all_contexts.csv")
    if os.path.exists(all_contexts_path):
        rows = load_hierarchical_csv(all_contexts_path)
        for row in rows:
            primary_context = row["contexts"][0] if row["contexts"] else None
            if primary_context:
                if primary_context not in label_map:
                    label_map[primary_context] = label_idx
                    label_idx += 1
                texts.append(row["message"])
                labels.append(label_map[primary_context])

    return texts, labels, label_map


def prepare_level2_data(data_dir, context):
    """
    Level 2: Entity classifier for a specific context (softmax).
    """
    texts, labels = [], []
    label_map = {}
    label_idx = 0

    ctx_path = os.path.join(data_dir, f"{context}.csv")
    if os.path.exists(ctx_path):
        rows = load_hierarchical_csv(ctx_path)
        for row in rows:
            # Use first entity for single-label training
            primary_entity = row["entities"][0] if row["entities"] else None
            if primary_entity:
                if primary_entity not in label_map:
                    label_map[primary_entity] = label_idx
                    label_idx += 1
                texts.append(row["message"])
                labels.append(label_map[primary_entity])

    return texts, labels, label_map


def prepare_level3_data(data_dir, context, entity):
    """
    Level 3: Sub-entity classifier for a specific entity (sigmoid multi-label).
    Returns binary vectors for multi-label classification.
    """
    # First, collect all possible sub-entities for this entity
    try:
        from training.constants import CONTEXT_REGISTRY
        entity_info = CONTEXT_REGISTRY[context]["entities"][entity]
        all_sub_entities = list(entity_info["sub_entities"].keys())
    except (ImportError, KeyError):
        # Fallback: collect from data
        all_sub_entities = set()
        ctx_path = os.path.join(data_dir, f"{context}.csv")
        if os.path.exists(ctx_path):
            rows = load_hierarchical_csv(ctx_path)
            for row in rows:
                if entity in row["entities"]:
                    all_sub_entities.update(row["sub_entities"])
        all_sub_entities = sorted(all_sub_entities)

    label_map = {se: i for i, se in enumerate(all_sub_entities)}
    num_labels = len(all_sub_entities)

    texts, labels = [], []
    ctx_path = os.path.join(data_dir, f"{context}.csv")
    if os.path.exists(ctx_path):
        rows = load_hierarchical_csv(ctx_path)
        for row in rows:
            # Only include rows that involve this entity
            if entity in row["entities"]:
                binary_vector = [0] * num_labels
                for se in row["sub_entities"]:
                    if se in label_map:
                        binary_vector[label_map[se]] = 1
                if sum(binary_vector) > 0:  # At least one label
                    texts.append(row["message"])
                    labels.append(binary_vector)

    return texts, labels, label_map


# ==============================================================================
# TRAINING
# ==============================================================================

def train_single_label(
    texts, labels, label_map,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    output_dir="models/level",
    epochs=10,
    batch_size=32,
    lr=2e-5,
    max_length=128,
    patience=3,
    device=None,
):
    """Train a softmax single-label classifier with early stopping."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Device: {device}")
    print(f"  Model: {model_name}")
    print(f"  Classes: {len(label_map)}")
    print(f"  Samples: {len(texts)}")

    # Filter out classes with fewer than 2 samples (can't stratify with 1)
    from collections import Counter
    label_counts = Counter(labels)
    rare_labels = {lbl for lbl, count in label_counts.items() if count < 2}
    if rare_labels:
        rare_names = [name for name, idx in label_map.items() if idx in rare_labels]
        print(f"  ⚠ Removing {len(rare_labels)} class(es) with <2 samples: {rare_names}")
        filtered = [(t, l) for t, l in zip(texts, labels) if l not in rare_labels]
        if not filtered:
            print("  ! No data left after filtering, skipping")
            return
        texts, labels = zip(*filtered)
        texts, labels = list(texts), list(labels)
        # Rebuild label_map with contiguous indices
        old_to_new = {}
        new_label_map = {}
        new_idx = 0
        for name, old_idx in sorted(label_map.items(), key=lambda x: x[1]):
            if old_idx not in rare_labels:
                old_to_new[old_idx] = new_idx
                new_label_map[name] = new_idx
                new_idx += 1
        labels = [old_to_new[l] for l in labels]
        label_map = new_label_map
        print(f"  Classes after filtering: {len(label_map)}, Samples: {len(texts)}")

    # Split data (use stratify only if all classes have >= 2 samples)
    label_counts = Counter(labels)
    can_stratify = all(c >= 2 for c in label_counts.values())
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42,
        stratify=labels if can_stratify else None
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  Early stopping patience: {patience} epochs")

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_map)
    ).to(device)

    # Datasets
    train_ds = SingleLabelDataset(X_train, y_train, tokenizer, max_length)
    val_ds = SingleLabelDataset(X_val, y_val, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # Training loop with early stopping
    best_f1 = 0
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        val_f1 = f1_score(all_labels, all_preds, average="weighted")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(output_dir, "final"))
            tokenizer.save_pretrained(os.path.join(output_dir, "final"))
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val F1: {val_f1:.4f} ★ best")
        else:
            epochs_without_improvement += 1
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val F1: {val_f1:.4f} (no improvement {epochs_without_improvement}/{patience})")

        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping: no improvement for {patience} epochs")
            break

    # Save label mappings
    reverse_map = {v: k for k, v in label_map.items()}
    with open(os.path.join(output_dir, "final", "label_mappings.json"), "w") as f:
        json.dump(reverse_map, f, indent=2)

    # Final report
    print(f"\n  Best Val F1: {best_f1:.4f}")
    print(f"  Stopped at epoch: {epoch+1}/{epochs}")
    print(f"  Saved to: {output_dir}/final")

    all_class_ids = sorted(set(all_labels) | set(all_preds))
    target_names = [reverse_map.get(i, f"class_{i}") for i in all_class_ids]
    print("\n" + classification_report(all_labels, all_preds, labels=all_class_ids, target_names=target_names))

    return best_f1


def train_multi_label(
    texts, labels, label_map,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    output_dir="models/level3",
    epochs=15,
    batch_size=32,
    lr=2e-5,
    max_length=128,
    threshold=0.5,
    patience=3,
    device=None,
):
    """Train a sigmoid multi-label classifier (Level 3) with early stopping."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_labels = len(label_map)
    print(f"\n  Device: {device}")
    print(f"  Model: {model_name}")
    print(f"  Labels (multi-label): {num_labels}")
    print(f"  Samples: {len(texts)}")
    print(f"  Threshold: {threshold}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  Early stopping patience: {patience} epochs")

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification"
    ).to(device)

    # Datasets
    train_ds = MultiLabelDataset(X_train, y_train, tokenizer, max_length)
    val_ds = MultiLabelDataset(X_val, y_val, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Loss function: BCEWithLogitsLoss for multi-label
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # Training loop with early stopping
    best_f1 = 0
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(outputs.logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                preds = (probs >= threshold).astype(int)
                all_preds.extend(preds)
                all_labels_v.extend(batch["labels"].numpy().astype(int))

        all_preds = np.array(all_preds)
        all_labels_v = np.array(all_labels_v)
        val_f1 = f1_score(all_labels_v, all_preds, average="micro")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(output_dir, "final"))
            tokenizer.save_pretrained(os.path.join(output_dir, "final"))
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val F1 (micro): {val_f1:.4f} ★ best")
        else:
            epochs_without_improvement += 1
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val F1 (micro): {val_f1:.4f} (no improvement {epochs_without_improvement}/{patience})")

        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping: no improvement for {patience} epochs")
            break

    # Save label mappings and config
    reverse_map = {v: k for k, v in label_map.items()}
    config = {
        "label_mappings": reverse_map,
        "threshold": threshold,
        "problem_type": "multi_label_classification",
    }
    with open(os.path.join(output_dir, "final", "label_mappings.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Best Val F1 (micro): {best_f1:.4f}")
    print(f"  Stopped at epoch: {epoch+1}/{epochs}")
    print(f"  Saved to: {output_dir}/final")

    # Per-label report
    reverse_map_list = [reverse_map[str(i)] if str(i) in reverse_map else reverse_map.get(i, f"label_{i}") for i in range(num_labels)]
    print("\n  Per-label F1:")
    for i, name in enumerate(reverse_map_list):
        if all_labels_v[:, i].sum() > 0:
            label_f1 = f1_score(all_labels_v[:, i], all_preds[:, i])
            print(f"    {name}: {label_f1:.4f} ({int(all_labels_v[:, i].sum())} samples)")

    return best_f1


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train hierarchical multi-label classifiers")
    parser.add_argument("--data-dir", type=str, default="training/data/hierarchical",
                        help="Directory with generated CSV files")
    parser.add_argument("--output-dir", type=str, default="training/models/hierarchical",
                        help="Base output directory for models")
    parser.add_argument("--level", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Train specific level only (default: all)")
    parser.add_argument("--context", type=str, default=None,
                        help="Context for Level 2/3 (required if --level 2 or 3)")
    parser.add_argument("--entity", type=str, default=None,
                        help="Entity for Level 3 (required if --level 3)")
    parser.add_argument("--model-name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Base model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for Level 3 multi-label (default: 0.5)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping: stop after N epochs without improvement (default: 3)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("HIERARCHICAL MULTI-LABEL CLASSIFIER TRAINING")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Base model: {args.model_name}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Early stopping patience: {args.patience} epochs")

    levels_to_train = [args.level] if args.level is not None else [0, 1, 2, 3]

    levels_to_train = [args.level] if args.level is not None else [0, 1, 2, 3]

    # =========================================================================
    # ROUTING: professional, learning, social, psychological, personal,
    #          rag_query, chitchat, off_topic
    # =========================================================================
    if 0 in levels_to_train:
        print(f"\n{'='*80}")
        print("ROUTING - professional, learning, social, psychological, personal, rag_query, chitchat, off_topic")
        print(f"{'='*80}")
        texts, labels, label_map = prepare_level0_data(args.data_dir)
        if texts:
            train_single_label(
                texts, labels, label_map,
                model_name=args.model_name,
                output_dir=os.path.join(args.output_dir, "routing"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                patience=args.patience,
                device=device,
            )
        else:
            print("  ! No data found for routing")

    # =========================================================================
    # CONTEXTS: professional, learning, social, psychological, personal
    # =========================================================================
    if 1 in levels_to_train:
        print(f"\n{'='*80}")
        print("CONTEXTS - professional, learning, social, psychological, personal")
        print(f"{'='*80}")
        texts, labels, label_map = prepare_level1_data(args.data_dir)
        if texts:
            train_single_label(
                texts, labels, label_map,
                model_name=args.model_name,
                output_dir=os.path.join(args.output_dir, "contexts"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                patience=args.patience,
                device=device,
            )
        else:
            print("  ! No data found for contexts")

    # =========================================================================
    # ENTITIES per context
    #   professional → current_position, professional_experience, awards, ...
    #   learning → current_skills, languages, education_history, ...
    #   etc.
    # =========================================================================
    if 2 in levels_to_train:
        contexts = [args.context] if args.context else ["professional", "learning", "social", "psychological", "personal"]
        for ctx in contexts:
            print(f"\n{'='*80}")
            print(f"{ctx.upper()} - ENTITIES")
            print(f"{'='*80}")
            texts, labels, label_map = prepare_level2_data(args.data_dir, ctx)
            if texts:
                train_single_label(
                    texts, labels, label_map,
                    model_name=args.model_name,
                    output_dir=os.path.join(args.output_dir, ctx, "entities"),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    max_length=args.max_length,
                    patience=args.patience,
                    device=device,
                )
            else:
                print(f"  ! No data found for {ctx} entities")

    # =========================================================================
    # SUB-ENTITIES per entity (sigmoid multi-label)
    #   professional > professional_aspirations → dream_roles, compensation_expectations, ...
    #   learning > current_skills → skills, proficiency, experience, ...
    #   etc.
    # =========================================================================
    if 3 in levels_to_train:
        try:
            from training.constants import CONTEXT_REGISTRY
            contexts = [args.context] if args.context else list(CONTEXT_REGISTRY.keys())
            for ctx in contexts:
                entities_to_train = [args.entity] if args.entity else list(CONTEXT_REGISTRY[ctx]["entities"].keys())
                for entity in entities_to_train:
                    entity_info = CONTEXT_REGISTRY[ctx]["entities"][entity]
                    num_subs = len(entity_info["sub_entities"])
                    if num_subs < 2:
                        print(f"\n  Skipping {ctx} > {entity} (only {num_subs} sub-entity, no classifier needed)")
                        continue

                    sub_names = ", ".join(entity_info["sub_entities"].keys())
                    print(f"\n{'='*80}")
                    print(f"{ctx.upper()} > {entity.upper()} - SUB-ENTITIES ({num_subs} labels, multi-label)")
                    print(f"  → {sub_names}")
                    print(f"{'='*80}")
                    texts, labels, label_map = prepare_level3_data(args.data_dir, ctx, entity)
                    if texts and len(texts) >= 20:
                        train_multi_label(
                            texts, labels, label_map,
                            model_name=args.model_name,
                            output_dir=os.path.join(args.output_dir, ctx, entity),
                            epochs=args.epochs + 5,  # More epochs for multi-label
                            batch_size=args.batch_size,
                            lr=args.lr,
                            max_length=args.max_length,
                            threshold=args.threshold,
                            patience=args.patience,
                            device=device,
                        )
                    else:
                        print(f"  ! Insufficient data for {ctx} > {entity} ({len(texts) if texts else 0} samples)")
        except ImportError:
            print("  ! Could not import CONTEXT_REGISTRY. Run with --context and --entity flags.")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\n  Models saved to: {args.output_dir}/")
    print("    routing/                                 - rag_query, chitchat, off_topic, professional, ...")
    print("    contexts/                                - professional, learning, social, psychological, personal")
    print("    professional/entities/                   - current_position, professional_experience, awards, ...")
    print("    professional/current_position/           - role, company, compensation, start_date")
    print("    professional/professional_aspirations/   - dream_roles, compensation_expectations, ...")
    print("    learning/entities/                       - current_skills, languages, education_history, ...")
    print("    ...etc")


if __name__ == "__main__":
    main()
