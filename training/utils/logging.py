from typing import List, Tuple
import pandas as pd
from training.constants import *

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

    print(f"\nOverall:")
    print(f"  Total messages: {len(df):,}")
    print(f"  Unique messages: {df['message'].nunique():,}")
    print(f"  Categories: {len(df['subcategory'].unique())}")

    print(f"\nBy subcategory:")
    for category_key in df["subcategory"].unique():
        # Try all category dictionaries
        category_name = (
            ASPIRATION_CATEGORIES.get(category_key, {}).get("name") or
            PROFESSIONAL_CATEGORIES.get(category_key, {}).get("name") or
            PSYCHOLOGICAL_CATEGORIES.get(category_key, {}).get("name") or
            LEARNING_CATEGORIES.get(category_key, {}).get("name") or
            SOCIAL_CATEGORIES.get(category_key, {}).get("name") or
            EMOTIONAL_CATEGORIES.get(category_key, {}).get("name") or
            RAG_QUERY_CATEGORIES.get(category_key, {}).get("name") or
            CHITCHAT_CATEGORIES.get(category_key, {}).get("name") or
            OFF_TOPIC_CATEGORIES.get(category_key, {}).get("name") or
            category_key
        )
        count = len(df[df["subcategory"] == category_key])
        category_type = df[df["subcategory"] == category_key].iloc[0]["category"]
        print(f"  [{category_type.upper()}] {category_name}: {count:,} messages")

    print(f"\nMessage length statistics:")
    df["length"] = df["message"].str.len()
    print(f"  Min: {df['length'].min()} chars")
    print(f"  Max: {df['length'].max()} chars")
    print(f"  Mean: {df['length'].mean():.1f} chars")
    print(f"  Median: {df['length'].median():.1f} chars")
