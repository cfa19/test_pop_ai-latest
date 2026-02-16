"""
Generate Hierarchical Multi-Label Training Data using OpenAI

New taxonomy (5 contexts + 3 non-context types):
1. PROFESSIONAL (6 entities, ~20 sub-entities)
2. LEARNING (11 entities, ~30 sub-entities)
3. SOCIAL (5 entities, ~25 sub-entities)
4. PSYCHOLOGICAL (10 entities, ~30 sub-entities)
5. PERSONAL (12 entities, ~50 sub-entities)
6. RAG_QUERY (8 subcategories) - flat
7. CHITCHAT (5 subcategories) - flat
8. OFF_TOPIC (5 subcategories) - flat

Generates three types of training data:
- Single-label: 1 message -> 1 context/entity/sub_entity (70%)
- Multi-label: 1 message -> multiple sub-entities within same context (20%)
- Cross-context: 1 message -> multiple contexts (10%)

Output CSV format:
  message, contexts, entities, sub_entities
  (pipe-separated for multi-label: "ctx1|ctx2", "ent1|ent2", "sub1|sub2|sub3")
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import get_openai
from training.constants import CONTEXT_REGISTRY, NON_CONTEXT_REGISTRY, ALL_CONTEXTS, ALL_NON_CONTEXTS, ALL_TYPES
from training.utils.generation import (
    generate_full_context,
    generate_non_context_messages,
    generate_messages_by_type,
    generate_cross_context_messages,
)


def save_hierarchical_csv(data: list, filepath: str):
    """Save hierarchical training data to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message", "contexts", "entities", "sub_entities"])
        for row in data:
            writer.writerow(row)
    print(f"\n  Saved {len(data)} rows to {filepath}")


def save_flat_csv(data: list, filepath: str):
    """Save flat (non-context) training data to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message", "category_type", "subcategory"])
        for row in data:
            writer.writerow(row)
    print(f"\n  Saved {len(data)} rows to {filepath}")


def show_taxonomy(context: str):
    """Print the taxonomy for a context."""
    entities = CONTEXT_REGISTRY[context]["entities"]
    total_subs = sum(len(e["sub_entities"]) for e in entities.values())
    print(f"\n  {context.upper()} ({len(entities)} entities, {total_subs} sub-entities):")
    for ek, ev in entities.items():
        subs = list(ev["sub_entities"].keys())
        print(f"    {ek}: {', '.join(subs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hierarchical multi-label training data"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="all",
        choices=ALL_TYPES + ["all", "all-contexts", "all-non-contexts"],
        help="Which context/type to generate (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/data/hierarchical",
        help="Output directory for CSV files (default: training/data/hierarchical)"
    )
    parser.add_argument(
        "--messages-per-sub-entity",
        type=int,
        default=25,
        help="Single-label messages per sub-entity (default: 25)"
    )
    parser.add_argument(
        "--multilabel-messages",
        type=int,
        default=50,
        help="Multi-label messages per context (default: 50)"
    )
    parser.add_argument(
        "--cross-context-messages",
        type=int,
        default=25,
        help="Cross-context messages per context pair (default: 25)"
    )
    parser.add_argument(
        "--non-context-messages",
        type=int,
        default=25,
        help="Messages per subcategory for non-context types (default: 25)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="OpenAI temperature (default: 0.8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Batch size for API calls (default: 25)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--show-taxonomy",
        action="store_true",
        help="Show the full taxonomy and exit"
    )

    args = parser.parse_args()

    # Show taxonomy and exit
    if args.show_taxonomy:
        print("=" * 80)
        print("HIERARCHICAL CLASSIFICATION TAXONOMY")
        print("=" * 80)
        for ctx in ALL_CONTEXTS:
            show_taxonomy(ctx)
        print(f"\n  Non-context types: {', '.join(ALL_NON_CONTEXTS)}")
        for nct in ALL_NON_CONTEXTS:
            cats = NON_CONTEXT_REGISTRY[nct]["categories"]
            print(f"    {nct}: {', '.join(cats.keys())}")
        return

    # Determine what to generate
    contexts_to_generate = []
    non_contexts_to_generate = []

    if args.context == "all":
        contexts_to_generate = ALL_CONTEXTS
        non_contexts_to_generate = ALL_NON_CONTEXTS
    elif args.context == "all-contexts":
        contexts_to_generate = ALL_CONTEXTS
    elif args.context == "all-non-contexts":
        non_contexts_to_generate = ALL_NON_CONTEXTS
    elif args.context in ALL_CONTEXTS:
        contexts_to_generate = [args.context]
    elif args.context in ALL_NON_CONTEXTS:
        non_contexts_to_generate = [args.context]

    print("=" * 80)
    print("HIERARCHICAL MULTI-LABEL TRAINING DATA GENERATION")
    print("=" * 80)

    # Show what will be generated
    if contexts_to_generate:
        print("\nContexts to generate:")
        for ctx in contexts_to_generate:
            show_taxonomy(ctx)

    if non_contexts_to_generate:
        print("\nNon-context types to generate:")
        for nct in non_contexts_to_generate:
            cats = NON_CONTEXT_REGISTRY[nct]["categories"]
            print(f"  {nct}: {', '.join(cats.keys())}")

    # Estimate total messages
    total_estimate = 0
    for ctx in contexts_to_generate:
        entities = CONTEXT_REGISTRY[ctx]["entities"]
        n_subs = sum(len(e["sub_entities"]) for e in entities.values())
        ctx_total = (n_subs * args.messages_per_sub_entity) + args.multilabel_messages + args.cross_context_messages
        total_estimate += ctx_total
    for nct in non_contexts_to_generate:
        cats = NON_CONTEXT_REGISTRY[nct]["categories"]
        total_estimate += len(cats) * args.non_context_messages

    print(f"\nConfiguration:")
    print(f"  Messages per sub-entity: {args.messages_per_sub_entity}")
    print(f"  Multi-label per context: {args.multilabel_messages}")
    print(f"  Cross-context per context: {args.cross_context_messages}")
    print(f"  Non-context per subcategory: {args.non_context_messages}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Model: {args.model}")
    print(f"  Estimated total: ~{total_estimate:,} messages")
    print(f"  Output: {args.output_dir}")

    # Initialize OpenAI
    client = get_openai()

    # Generate context data (hierarchical)
    all_hierarchical = []
    for ctx in contexts_to_generate:
        results = generate_full_context(
            client,
            context=ctx,
            messages_per_sub_entity=args.messages_per_sub_entity,
            multilabel_messages=args.multilabel_messages,
            cross_context_messages=args.cross_context_messages,
            temperature=args.temperature,
            batch_size=args.batch_size,
            model=args.model,
        )

        # Collect all results
        context_data = []
        for msg, ctx_label, entity, sub_ents in results["single_label"]:
            context_data.append((msg, ctx_label, entity, sub_ents))
        for msg, ctx_label, entities_str, sub_ents_str in results["multi_label"]:
            context_data.append((msg, ctx_label, entities_str, sub_ents_str))
        for msg, ctxs_str, entities_str, sub_ents_str in results["cross_context"]:
            context_data.append((msg, ctxs_str, entities_str, sub_ents_str))

        # Save per-context CSV
        save_hierarchical_csv(context_data, os.path.join(args.output_dir, f"{ctx}.csv"))
        all_hierarchical.extend(context_data)

    # Save combined hierarchical CSV
    if all_hierarchical:
        save_hierarchical_csv(all_hierarchical, os.path.join(args.output_dir, "all_contexts.csv"))

    # Generate non-context data (flat)
    for nct in non_contexts_to_generate:
        messages = generate_messages_by_type(
            client,
            category_type=nct,
            messages_per_category=args.non_context_messages,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
        save_flat_csv(messages, os.path.join(args.output_dir, f"{nct}.csv"))

    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n  Hierarchical messages: {len(all_hierarchical):,}")
    print(f"  Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review the generated data in the CSV files")
    print("  2. Train on vast.ai: python training/scripts/train_multilabel.py")


if __name__ == "__main__":
    main()
