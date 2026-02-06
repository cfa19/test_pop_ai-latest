"""
Generate Aspirational Context Training Data using OpenAI

Uses OpenAI to generate aspirations in 6 categories:
1. Dream Roles - Career positions and titles
2. Salary Expectations - Compensation and financial goals
3. Life Goals Beyond Career - Personal life, family, lifestyle
4. Values - Work principles, work-life balance, ethics
5. Impact & Legacy - Making a difference, leaving a mark
6. Skill & Expertise - Mastering technologies, becoming an expert

Each category gets its own patterns and aspirations.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import get_openai
from training.constants.aspirational import (
    ASPIRATION_CATEGORIES,
    ASPIRATION_GENERATION_SYSTEM_PROMPT,
    PATTERN_GENERATION_SYSTEM_PROMPT,
    build_aspiration_generation_prompt,
    build_pattern_generation_prompt,
)
from training.utils.aspirational import _batch_generate_with_openai

# ==============================================================================
# PATTERN GENERATION
# ==============================================================================


def generate_aspirational_patterns(client, category_key: str, num_patterns: int = 20, temperature: float = 0.8, batch_size: int = 10) -> List[str]:
    """
    Generate category-specific message patterns with [OBJ] placeholders.

    Args:
        client: OpenAI client
        category_key: Key from ASPIRATION_CATEGORIES
        num_patterns: Number of patterns to generate
        temperature: Creativity level (0.0-1.0)
        batch_size: Number of patterns per API call

    Returns:
        List of pattern strings with [OBJ] placeholder
    """
    category = ASPIRATION_CATEGORIES[category_key]
    print(f"\nGenerating {num_patterns} patterns for {category['name']}...")

    # Validator to ensure patterns contain [OBJ] placeholder
    def validate_pattern(pattern: str) -> bool:
        return "[OBJ]" in pattern

    return _batch_generate_with_openai(
        client=client,
        category_key=category_key,
        total_count=num_patterns,
        batch_size=batch_size,
        temperature=temperature,
        prompt_builder_fn=build_pattern_generation_prompt,
        system_prompt=PATTERN_GENERATION_SYSTEM_PROMPT,
        result_key="patterns",
        item_name="patterns",
        validator_fn=validate_pattern,
    )


# ==============================================================================
# OBJECT GENERATION
# ==============================================================================


def generate_aspirational_objects(client, category_key: str, num_aspirations: int = 30, temperature: float = 0.8, batch_size: int = 10) -> List[str]:
    """
    Generate aspirations for a specific category.

    Args:
        client: OpenAI client
        category_key: Key from ASPIRATION_CATEGORIES
        num_aspirations: Number of aspirations to generate
        temperature: Creativity level (0.0-1.0)
        batch_size: Number of aspirations per API call

    Returns:
        List of aspiration strings
    """
    category = ASPIRATION_CATEGORIES[category_key]
    print(f"\nGenerating {num_aspirations} aspirations for {category['name']}...")

    return _batch_generate_with_openai(
        client=client,
        category_key=category_key,
        total_count=num_aspirations,
        batch_size=batch_size,
        temperature=temperature,
        prompt_builder_fn=build_aspiration_generation_prompt,
        system_prompt=ASPIRATION_GENERATION_SYSTEM_PROMPT,
        result_key="aspirations",
        item_name="aspirations",
        validator_fn=None,  # No validation needed for aspirations
    )


# ==============================================================================
# BATCH GENERATION BY CATEGORY
# ==============================================================================


def generate_all_categories(
    client,
    patterns_per_category: int = 20,
    aspirations_per_category: int = 30,
    temperature: float = 0.8,
    categories: List[str] = None,
    batch_size: int = 10,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate patterns and aspirations for all (or selected) categories.

    Args:
        client: OpenAI client
        patterns_per_category: Number of patterns per category
        aspirations_per_category: Number of aspirations per category
        temperature: Creativity level
        categories: List of category keys to generate (None = all)

    Returns:
        Dictionary mapping category_key -> {patterns: [...], aspirations: [...]}
    """
    if categories is None:
        categories = list(ASPIRATION_CATEGORIES.keys())

    results = {}

    for i, category_key in enumerate(categories, 1):
        category_name = ASPIRATION_CATEGORIES[category_key]["name"]
        print(f"\n{'=' * 80}")
        print(f"Category {i}/{len(categories)}: {category_name}")
        print(f"{'=' * 80}")

        # Generate patterns
        patterns = generate_aspirational_patterns(
            client, category_key, num_patterns=patterns_per_category, temperature=temperature, batch_size=batch_size
        )

        # Wait for rate limit
        time.sleep(2)

        # Generate aspirations
        aspirations = generate_aspirational_objects(
            client, category_key, num_aspirations=aspirations_per_category, temperature=temperature, batch_size=batch_size
        )

        results[category_key] = {"patterns": patterns, "aspirations": aspirations}

        # Wait before next category
        if i < len(categories):
            print("  Waiting 2s before next category...")
            time.sleep(2)

    return results


# ==============================================================================
# COMBINATION
# ==============================================================================


def combine_patterns_and_aspirations(
    category_data: Dict[str, Dict[str, List[str]]], max_samples_per_category: int = None
) -> List[Tuple[str, str, str]]:
    """
    Combine patterns and aspirations within each category.

    Args:
        category_data: Dict of category -> {patterns, aspirations}
        max_samples_per_category: Max samples per category (None = all)

    Returns:
        List of (message, category="aspirational", subcategory) tuples
    """
    print(f"\n{'=' * 80}")
    print("COMBINING PATTERNS AND ASPIRATIONS")
    print(f"{'=' * 80}")

    all_messages = []

    for category_key, data in category_data.items():
        category_name = ASPIRATION_CATEGORIES[category_key]["name"]
        patterns = data["patterns"]
        aspirations = data["aspirations"]

        print(f"\n{category_name}:")
        print(f"  Patterns: {len(patterns)}")
        print(f"  Aspirations: {len(aspirations)}")
        print(f"  Possible combinations: {len(patterns) * len(aspirations):,}")

        count = 0
        for pattern in patterns:
            for aspiration in aspirations:
                message = pattern.replace("[OBJ]", aspiration)
                all_messages.append((message, "aspirational", category_key))
                count += 1

                # Early exit if max reached
                if max_samples_per_category and count >= max_samples_per_category:
                    print(f"  ✓ Limited to {max_samples_per_category} samples")
                    break

            if max_samples_per_category and count >= max_samples_per_category:
                break

        if not max_samples_per_category:
            print(f"  ✓ Generated {count:,} messages")

    print(f"\n{'=' * 80}")
    print(f"Total messages: {len(all_messages):,}")
    print(f"{'=' * 80}")

    return all_messages


# ==============================================================================
# SAVE FUNCTIONS
# ==============================================================================


def save_to_csv(messages: List[Tuple[str, str, str]], output_path: str, category_data: Dict[str, Dict[str, List[str]]] = None):
    """
    Save messages to CSV file with subcategory column.

    Args:
        messages: List of (message, category, subcategory) tuples
        output_path: Path to save CSV file
        category_data: Optional category data to save separately
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save main dataset
    df = pd.DataFrame(messages, columns=["message", "category", "subcategory"])
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Saved {len(messages):,} messages to: {output_path}")

    # Save category components if provided
    if category_data:
        components_dir = output_path.replace(".csv", "_components")
        os.makedirs(components_dir, exist_ok=True)

        for category_key, data in category_data.items():
            category_name = ASPIRATION_CATEGORIES[category_key]["name"]

            # Save patterns
            patterns_path = os.path.join(components_dir, f"{category_key}_patterns.txt")
            with open(patterns_path, "w", encoding="utf-8") as f:
                for pattern in data["patterns"]:
                    f.write(pattern + "\n")

            # Save aspirations
            aspirations_path = os.path.join(components_dir, f"{category_key}_aspirations.txt")
            with open(aspirations_path, "w", encoding="utf-8") as f:
                for aspiration in data["aspirations"]:
                    f.write(aspiration + "\n")

        print(f"✓ Saved component files to: {components_dir}/")

    # Print sample by category
    print("\nSample messages by category:")
    for category_key in df["subcategory"].unique():
        category_name = ASPIRATION_CATEGORIES[category_key]["name"]
        samples = df[df["subcategory"] == category_key].head(3)
        print(f"\n{category_name}:")
        for i, row in samples.iterrows():
            print(f"  - {row['message']}")


def print_statistics(messages: List[Tuple[str, str, str]], category_data: Dict[str, Dict[str, List[str]]]):
    """Print generation statistics."""
    print("\n" + "=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)

    df = pd.DataFrame(messages, columns=["message", "category", "subcategory"])

    # Overall stats
    print("\nOverall:")
    print(f"  Total messages: {len(df):,}")
    print(f"  Unique messages: {df['message'].nunique():,}")
    print(f"  Categories: {len(category_data)}")

    # Per-category stats
    print("\nBy subcategory:")
    for category_key, data in category_data.items():
        category_name = ASPIRATION_CATEGORIES[category_key]["name"]
        count = len(df[df["subcategory"] == category_key])
        patterns = len(data["patterns"])
        aspirations = len(data["aspirations"])
        print(f"  {category_name}:")
        print(f"    Patterns: {patterns}")
        print(f"    Aspirations: {aspirations}")
        print(f"    Messages: {count:,}")

    # Message length stats
    print("\nMessage length statistics:")
    df["length"] = df["message"].str.len()
    print(f"  Min: {df['length'].min()} chars")
    print(f"  Max: {df['length'].max()} chars")
    print(f"  Mean: {df['length'].mean():.1f} chars")
    print(f"  Median: {df['length'].median():.1f} chars")


# ==============================================================================
# LOAD FUNCTIONS
# ==============================================================================


def load_category_data(components_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load previously saved category components.

    Args:
        components_dir: Directory containing component files

    Returns:
        Dictionary mapping category_key -> {patterns, aspirations}
    """
    print(f"\nLoading category components from: {components_dir}")

    category_data = {}

    for category_key in ASPIRATION_CATEGORIES.keys():
        patterns_path = os.path.join(components_dir, f"{category_key}_patterns.txt")
        aspirations_path = os.path.join(components_dir, f"{category_key}_aspirations.txt")

        if os.path.exists(patterns_path) and os.path.exists(aspirations_path):
            with open(patterns_path, "r", encoding="utf-8") as f:
                patterns = [line.strip() for line in f if line.strip()]

            with open(aspirations_path, "r", encoding="utf-8") as f:
                aspirations = [line.strip() for line in f if line.strip()]

            category_data[category_key] = {"patterns": patterns, "aspirations": aspirations}

            category_name = ASPIRATION_CATEGORIES[category_key]["name"]
            print(f"  ✓ {category_name}: {len(patterns)} patterns, {len(aspirations)} aspirations")

    if not category_data:
        print("  ⚠ No category components found")

    return category_data


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate aspirational training data using OpenAI with 6 categories")
    parser.add_argument("--output", type=str, default="training/data/aspirational_messages.csv", help="Output CSV file path")
    parser.add_argument("--patterns-per-category", type=int, default=20, help="Number of patterns per category (default: 20)")
    parser.add_argument("--aspirations-per-category", type=int, default=30, help="Number of aspirations per category (default: 30)")
    parser.add_argument("--max-samples-per-category", type=int, help="Maximum samples per category (default: all combinations)")
    parser.add_argument("--temperature", type=float, default=0.8, help="OpenAI temperature (default: 0.8)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generating patterns and aspirations (default: 10)")
    parser.add_argument("--save-components", action="store_true", help="Save patterns and aspirations by category")
    parser.add_argument("--load-components", type=str, help="Load components from directory instead of generating")
    parser.add_argument("--categories", nargs="+", choices=list(ASPIRATION_CATEGORIES.keys()), help="Specific categories to generate (default: all)")

    args = parser.parse_args()

    print("=" * 80)
    print("ASPIRATIONAL DATA GENERATION (6 Categories)")
    print("=" * 80)

    # Show categories
    print("\nCategories:")
    for key, info in ASPIRATION_CATEGORIES.items():
        print(f"  {key}: {info['name']}")

    print("\nConfiguration:")
    print(f"  Output: {args.output}")
    print(f"  Patterns per category: {args.patterns_per_category}")
    print(f"  Aspirations per category: {args.aspirations_per_category}")
    print(f"  Max samples per category: {args.max_samples_per_category or 'All combinations'}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Categories: {args.categories or 'All (6)'}")

    # Load or generate category data
    if args.load_components:
        category_data = load_category_data(args.load_components)
        if not category_data:
            print("✗ Failed to load components. Exiting.")
            return
    else:
        # Initialize OpenAI client
        client = get_openai()

        # Generate all categories
        category_data = generate_all_categories(
            client,
            patterns_per_category=args.patterns_per_category,
            aspirations_per_category=args.aspirations_per_category,
            temperature=args.temperature,
            categories=args.categories,
            batch_size=args.batch_size,
        )

    if not category_data:
        print("✗ No data generated. Exiting.")
        return

    # Combine patterns and aspirations
    messages = combine_patterns_and_aspirations(category_data, max_samples_per_category=args.max_samples_per_category)

    # Print statistics
    print_statistics(messages, category_data)

    # Save to CSV
    save_to_csv(messages, args.output, category_data=category_data if args.save_components else None)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated data")
    print("  2. Combine with other category data")
    print("  3. Train the intent classifier model")

    if args.save_components:
        components_dir = args.output.replace(".csv", "_components")
        print(f"\nGenerated component files in: {components_dir}/")
        print("You can reuse these with --load-components")


if __name__ == "__main__":
    main()
