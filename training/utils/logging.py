from typing import List, Tuple

import pandas as pd

from training.constants import (
    CHITCHAT_CATEGORIES,
    CONTEXT_REGISTRY,
    OFF_TOPIC_CATEGORIES,
    RAG_QUERY_CATEGORIES,
)

# Build flat lookup: entity_key -> display name
_CATEGORY_NAMES = {}
for _ctx_data in CONTEXT_REGISTRY.values():
    for _key, _entity in _ctx_data["entities"].items():
        _CATEGORY_NAMES[_key] = _entity["name"]
for _cats in [RAG_QUERY_CATEGORIES, CHITCHAT_CATEGORIES, OFF_TOPIC_CATEGORIES]:
    for _key, _cat in _cats.items():
        _CATEGORY_NAMES[_key] = _cat["name"]


def print_statistics(messages: List[Tuple]):
    """Print generation statistics."""
    print("\n" + "=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)

    has_entities = len(messages[0]) == 4
    if has_entities:
        df = pd.DataFrame(
            [(msg, cat, subcat) for msg, cat, subcat, _ in messages],
            columns=["message", "category", "subcategory"]
        )
    else:
        df = pd.DataFrame(messages, columns=["message", "category", "subcategory"])

    print("\nOverall:")
    print(f"  Total messages: {len(df):,}")
    print(f"  Unique messages: {df['message'].nunique():,}")
    print(f"  Categories: {len(df['subcategory'].unique())}")

    print("\nBy subcategory:")
    for category_key in df["subcategory"].unique():
        category_name = _CATEGORY_NAMES.get(category_key, category_key)
        count = len(df[df["subcategory"] == category_key])
        category_type = df[df["subcategory"] == category_key].iloc[0]["category"]
        print(f"  [{category_type.upper()}] {category_name}: {count:,} messages")

    print("\nMessage length statistics:")
    df["length"] = df["message"].str.len()
    print(f"  Min: {df['length'].min()} chars")
    print(f"  Max: {df['length'].max()} chars")
    print(f"  Mean: {df['length'].mean():.1f} chars")
    print(f"  Median: {df['length'].median():.1f} chars")
