"""
Generate Training Data using OpenAI

Supports nine category types:
1. ASPIRATIONAL (6 subcategories):
   - Dream Roles, Salary Expectations, Life Goals, Values, Impact & Legacy, Skill & Expertise

2. PROFESSIONAL (4 subcategories):
   - Experiences, Skills, Certifications, Current Position

3. PSYCHOLOGICAL (4 subcategories):
   - Personality Profile, Values Hierarchy, Core Motivations, Working Styles

4. LEARNING (3 subcategories):
   - Knowledge, Learning Velocity, Preferred Learning Format

5. SOCIAL (4 subcategories):
   - Mentors, Journey Peers, People Helped, Testimonials

6. EMOTIONAL (4 subcategories):
   - Confidence, Energy Patterns, Stress Triggers, Celebration Moments

7. RAG_QUERY (8 subcategories):
   - Company Overview, Products & Services, Runners System, Programs & Pricing,
   - Credits System, Philosophy & Ethics, Transformation Index, Canonical Profile & Contexts

8. CHITCHAT (5 subcategories):
   - Greetings, Thanks & Appreciation, Farewells, Acknowledgments, Small Talk

9. OFF_TOPIC (5 subcategories):
   - Random Topics, Personal Life (Non-Career), General Knowledge Questions,
   - Nonsensical Messages, Current Events & News

Generates natural messages directly (no separate patterns/objects).
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import get_openai
from training.constants import *
from training.utils import generate_messages_by_type, print_statistics, save_to_csv

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using OpenAI (aspirational, professional, psychological, learning, social, emotional, rag_query, chitchat, and/or off_topic)"
    )
    parser.add_argument(
        "--category-type",
        type=str,
        default="rag_query",
        choices=["aspirational", "professional", "psychological", "learning", "social", "emotional", "rag_query", "chitchat", "off_topic", "all"],
        help="Type of messages to generate (default: chitchat)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/generated_messages.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--messages-per-category",
        type=int,
        nargs="+",
        default=[25],
        metavar="N",
        help="Number of messages per category. For --category-type all: use 9 values (one per type: aspirational, professional, psychological, learning, social, emotional, rag_query, chitchat, off_topic) or 1 value for all. For a single type: 1 value (default: 25)."
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
        "--categories",
        nargs="+",
        help="Specific subcategories to generate (default: all for selected type)"
    )

    args = parser.parse_args()

    # Validate --messages-per-category length
    n = len(args.messages_per_category)
    if args.category_type == "all":
        if n != 1 and n != 9:
            parser.error(
                "--messages-per-category: for --category-type all use 1 value (same for all types) or 9 values "
                "(one per type: aspirational, professional, psychological, learning, social, emotional, rag_query, chitchat, off_topic). "
                f"Got {n} value(s)."
            )
    else:
        if n == 0:
            parser.error("--messages-per-category: at least one value required.")
        if n > 1:
            # Use first value for single type (ignore rest)
            args.messages_per_category = [args.messages_per_category[0]]

    # Determine which category types to generate
    types_to_generate = []
    if args.category_type == "all":
        types_to_generate = ["aspirational", "professional", "psychological", "learning", "social", "emotional", "rag_query", "chitchat", "off_topic"]
    else:
        types_to_generate = [args.category_type]

    print("=" * 80)
    print("TRAINING DATA GENERATION")
    print("=" * 80)

    # Show categories for selected types
    print("\nCategories to generate:")
    for cat_type in types_to_generate:
        if cat_type == "aspirational":
            print("\n  ASPIRATIONAL (6 subcategories):")
            for key, info in ASPIRATION_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "professional":
            print("\n  PROFESSIONAL (4 subcategories):")
            for key, info in PROFESSIONAL_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "psychological":
            print("\n  PSYCHOLOGICAL (4 subcategories):")
            for key, info in PSYCHOLOGICAL_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "learning":
            print("\n  LEARNING (3 subcategories):")
            for key, info in LEARNING_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "social":
            print("\n  SOCIAL (4 subcategories):")
            for key, info in SOCIAL_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "emotional":
            print("\n  EMOTIONAL (4 subcategories):")
            for key, info in EMOTIONAL_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "rag_query":
            print("\n  RAG_QUERY (8 subcategories):")
            for key, info in RAG_QUERY_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "chitchat":
            print("\n  CHITCHAT (5 subcategories):")
            for key, info in CHITCHAT_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")
        elif cat_type == "off_topic":
            print("\n  OFF_TOPIC (5 subcategories):")
            for key, info in OFF_TOPIC_CATEGORIES.items():
                print(f"    - {key}: {info['name']}")

    print("\nConfiguration:")
    print(f"  Category type: {args.category_type}")
    print(f"  Output: {args.output}")
    if args.category_type == "all" and len(args.messages_per_category) == 9:
        print(f"  Messages per category (by type): {dict(zip(types_to_generate, args.messages_per_category))}")
    else:
        print(f"  Messages per category: {args.messages_per_category}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Specific subcategories: {args.categories or 'All'}")

    # Initialize OpenAI client
    client = get_openai()

    # Generate messages for each type (use per-type count when list has 9 elements)
    all_messages = []
    for i, cat_type in enumerate(types_to_generate):
        count = args.messages_per_category[i] if len(args.messages_per_category) > 1 else args.messages_per_category[0]
        messages = generate_messages_by_type(
            client,
            category_type=cat_type,
            messages_per_category=count,
            temperature=args.temperature,
            categories=args.categories,
            batch_size=args.batch_size
        )
        all_messages.extend(messages)

    messages = all_messages

    if not messages:
        print("✗ No messages generated. Exiting.")
        return

    # Print statistics
    print_statistics(messages)

    # Save to CSV
    save_to_csv(messages, os.path.join(args.output, f"{args.category_type}.csv"))

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated data")
    print("  2. Combine with other category data")
    print("  3. Train the intent classifier model")


if __name__ == "__main__":
    main()
