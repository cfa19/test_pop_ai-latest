import json
import os
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
        print("  Columns: message, category, subcategory, entities (JSON)")

    # Print sample by category
    print("\nSample messages by category:")
    for category_key in df["subcategory"].unique():
        category_name = _CATEGORY_NAMES.get(category_key, category_key)
        samples = df[df["subcategory"] == category_key].head(3)
        category_type = df[df["subcategory"] == category_key].iloc[0]["category"]
        print(f"\n[{category_type.upper()}] {category_name}:")
        for i, row in samples.iterrows():
            print(f"  - {row['message']}")
