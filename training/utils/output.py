import os
import pandas as pd
from typing import List, Tuple
import json

from training.constants import *


# ==============================================================================
# SAVE FUNCTIONS
# ==============================================================================

def save_to_csv(
    messages: List[Tuple],
    output_path: str,
):
    """
    Save messages to CSV file with subcategory and optional entities columns.

    Args:
        messages: List of (message, category, subcategory, entities?) tuples
        output_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    has_entities = len(messages[0]) == 4

    if has_entities:
        data = []
        for message, category, subcategory, entities in messages:
            entities_json = json.dumps(entities) if entities else "{}"
            data.append((message, category, subcategory, entities_json))
        df = pd.DataFrame(data, columns=["message", "category", "subcategory", "entities"])
    else:
        df = pd.DataFrame(messages, columns=["message", "category", "subcategory"])

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Saved {len(messages):,} messages to: {output_path}")

    if has_entities:
        print(f"  Columns: message, category, subcategory, entities (JSON)")

    # Print sample by category
    print("\nSample messages by category:")
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
        samples = df[df["subcategory"] == category_key].head(3)
        category_type = df[df["subcategory"] == category_key].iloc[0]["category"]
        print(f"\n[{category_type.upper()}] {category_name}:")
        for i, row in samples.iterrows():
            print(f"  - {row['message']}")
