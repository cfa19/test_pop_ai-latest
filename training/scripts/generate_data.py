"""
Generate Synthetic Training Data

Uses GPT-3.5-turbo to generate synthetic conversation messages for training:
1. Conversational chitchat (greetings, chitchat, unclear messages)
2. RAG queries based on content in general_embeddings_1024
3. Messages for each of the 6 Store A contexts
"""

import argparse
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import get_openai, get_supabase
from training.constants.categories import STORE_A_CONTEXTS
from training.utils.generation import generate_messages, generate_rag_queries
from training.utils.supabase import fetch_knowledge_base_content


def generate_balanced_dataset(
    output_path: str,
    rag_count: int = 100,
    context_count_per_category: int = 80,
    chitchat_count: int = 60,
    offtopic_count: int = 100,
    use_knowledge_base: bool = True,
):
    """
    Generate a balanced synthetic dataset for training.

    Args:
        output_path: Path to save CSV
        rag_count: Number of RAG query messages
        context_count_per_category: Number of messages per Store A context (6 categories)
        chitchat_count: Number of chitchat messages
        offtopic_count: Number of off-topic messages (for semantic gate testing, NOT classifier training)
        use_knowledge_base: Whether to use knowledge base for RAG queries

    Note:
        Off-topic messages are used for testing the semantic gate (Stage 1) which filters
        out off-topic messages. They are NOT used for training the BERT classifier (Stage 2).
    """
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 60)

    total_expected = rag_count + (context_count_per_category * 6) + chitchat_count + offtopic_count
    print(f"\nTarget dataset size: {total_expected} messages")
    print(f"  - RAG queries: {rag_count}")
    print(f"  - Professional: {context_count_per_category}")
    print(f"  - Psychological: {context_count_per_category}")
    print(f"  - Learning: {context_count_per_category}")
    print(f"  - Social: {context_count_per_category}")
    print(f"  - Emotional: {context_count_per_category}")
    print(f"  - Aspirational: {context_count_per_category}")
    print(f"  - Chitchat: {chitchat_count}")
    print(f"  - Off-topic: {offtopic_count} (for semantic gate testing)")
    print("\nNOTE: Off-topic messages are for testing the semantic gate,")
    print("      NOT for training the BERT classifier.")

    openai_client = get_openai()

    # Fetch knowledge base if needed
    knowledge_base = []
    if use_knowledge_base:
        try:
            supabase = get_supabase()
            knowledge_base = fetch_knowledge_base_content(supabase, limit=50)
        except Exception as e:
            print(f"Warning: Could not fetch knowledge base: {e}")
            print("Proceeding with generic RAG queries")

    # Generate RAG queries (rate limiting handled internally)
    if rag_count > 0:
        rag_data = generate_rag_queries(openai_client, knowledge_base, rag_count)
        with open(os.path.join(os.path.dirname(output_path), "rag_queries.txt"), "w", encoding="utf-8") as f:
            for message in rag_data:
                f.write(message + "\n")

    # Generate messages for each Store A context
    for category in STORE_A_CONTEXTS:
        if context_count_per_category > 0:
            context_data = generate_messages(openai_client, category, context_count_per_category)
            with open(os.path.join(os.path.dirname(output_path), f"{category}.txt"), "w", encoding="utf-8") as f:
                for message in context_data:
                    f.write(message + "\n")
            time.sleep(1)  # Rate limit pause

    # Generate chitchat messages
    if chitchat_count > 0:
        chitchat_data = generate_messages(openai_client, "chitchat", chitchat_count)
        with open(os.path.join(os.path.dirname(output_path), "chitchat.txt"), "w", encoding="utf-8") as f:
            for message in chitchat_data:
                f.write(message + "\n")

    # Generate off-topic messages (for semantic gate testing)
    if offtopic_count > 0:
        print("\nGenerating off-topic messages for semantic gate testing...")
        offtopic_data = generate_messages(openai_client, "off_topic", offtopic_count)
        with open(os.path.join(os.path.dirname(output_path), "off_topic.txt"), "w", encoding="utf-8") as f:
            for message in offtopic_data:
                f.write(message + "\n")

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print("\nReminder: off_topic messages are for testing the semantic gate,")
    print("          NOT for training the BERT classifier.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data using GPT-3.5")
    parser.add_argument("--output", type=str, default="data/processed/synthetic_labeled.csv", help="Output CSV path")
    parser.add_argument("--rag-count", type=int, default=0, help="Number of RAG query messages to generate")
    parser.add_argument("--context-count", type=int, default=0, help="Number of messages per Store A context (6 categories)")
    parser.add_argument("--chitchat-count", type=int, default=0, help="Number of conversational chitchat messages")
    parser.add_argument(
        "--offtopic-count", type=int, default=5000, help="Number of off-topic messages for semantic gate testing (NOT for classifier training)"
    )
    parser.add_argument("--no-knowledge-base", action="store_true", help="Skip fetching knowledge base content for RAG queries")

    args = parser.parse_args()

    generate_balanced_dataset(
        output_path=args.output,
        rag_count=args.rag_count,
        context_count_per_category=args.context_count,
        chitchat_count=args.chitchat_count,
        offtopic_count=args.offtopic_count,
        use_knowledge_base=not args.no_knowledge_base,
    )

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Review the generated data:")
    print(f"   cat {args.output}")
    print("\n2. Split into train/val/test:")
    print(f"   python scripts/prepare_data.py --task split --input {args.output} --output data/processed/intent/")
    print("\n3. Train the model:")
    print("   python scripts/train_intent_classifier.py --train-data data/processed/intent/train.csv --val-data data/processed/intent/val.csv")
    print("\n" + "=" * 60)
