"""
Generate Aspirational Training Data with Linguistic Patterns

Enhanced version that generates aspirations with both:
1. Content categories (WHAT): dream_roles, salary, life_goals, values, impact, skills
2. Linguistic patterns (HOW): explicit, future, intentional, identity, regret, conditional

This creates a richer dataset with multi-dimensional labeling.
"""

import argparse
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import get_openai
from training.constants.aspiration_patterns import (
    PATTERN_DISTRIBUTION,
    get_all_pattern_keys,
    get_pattern_commitment_level,
    get_pattern_name,
    get_pattern_templates,
    get_recommended_sample_count,
    validate_pattern_coverage,
)
from training.constants.aspirational import get_all_category_keys, get_category_name
from training.utils.esco_loader import load_esco_aspirations

# ==============================================================================
# ASPIRATIONAL OBJECT GENERATION
# ==============================================================================


def generate_aspirational_objects(client, category_key: str, num_aspirations: int = 30, temperature: float = 0.8) -> List[str]:
    """
    Generate aspirations for a specific content category using OpenAI.

    Args:
        client: OpenAI client
        category_key: Key from ASPIRATION_CATEGORIES
        num_aspirations: Number of aspirations to generate
        temperature: Creativity level (0.0-1.0)

    Returns:
        List of aspiration strings (verb phrases)
    """
    from training.scripts.generate_aspirational_data import generate_aspirational_objects as gen_objs

    return gen_objs(client, category_key, num_aspirations, temperature)


def combine_openai_and_esco_aspirations(
    openai_aspirations: Dict[str, List[str]], esco_occupation_limit: int = 500, esco_skill_limit: int = 500, esco_weight: float = 0.5
) -> Dict[str, List[str]]:
    """
    Combine OpenAI-generated aspirations with ESCO dataset aspirations.

    Args:
        openai_aspirations: Dict of category_key -> list of OpenAI aspirations
        esco_occupation_limit: Max ESCO occupations to load
        esco_skill_limit: Max ESCO skills to load
        esco_weight: Proportion of ESCO data to use (0.0-1.0)
                     0.5 = 50% OpenAI, 50% ESCO

    Returns:
        Dictionary of category_key -> combined aspirations list
    """
    print(f"\n{'=' * 80}")
    print("COMBINING OPENAI AND ESCO ASPIRATIONS")
    print(f"{'=' * 80}")
    print(f"ESCO weight: {esco_weight * 100:.0f}% ESCO, {(1 - esco_weight) * 100:.0f}% OpenAI")

    # Load ESCO aspirations
    try:
        esco_data = load_esco_aspirations(
            occupation_limit=esco_occupation_limit,
            skill_limit=esco_skill_limit,
            include_alternatives=True,
            skill_reuse_levels=["transversal", "cross-sectoral"],  # Focus on broadly applicable skills
            isco_groups=["1", "2", "3"],  # Managers, Professionals, Technicians
        )
    except FileNotFoundError as e:
        print(f"\n⚠ Warning: Could not load ESCO data: {e}")
        print("  Continuing with OpenAI-only aspirations")
        return openai_aspirations

    # Combine with each category
    combined = {}

    for category_key, openai_asps in openai_aspirations.items():
        category_name = get_category_name(category_key)

        # Determine which ESCO aspirations to use based on category
        if category_key in ["dream_roles", "skill_expertise"]:
            # Use ESCO occupations for roles, skills for expertise
            if category_key == "dream_roles":
                esco_asps = esco_data["occupations"]
            else:  # skill_expertise
                esco_asps = esco_data["skills"]
        else:
            # For other categories, use a mix of both
            esco_asps = esco_data["all"]

        # Calculate how many from each source
        total_needed = len(openai_asps)
        num_esco = int(total_needed * esco_weight)
        num_openai = total_needed - num_esco

        # Sample from each source
        import random

        esco_sample = random.sample(esco_asps, min(num_esco, len(esco_asps)))
        openai_sample = random.sample(openai_asps, min(num_openai, len(openai_asps)))

        # Combine and shuffle
        combined_asps = esco_sample + openai_sample
        random.shuffle(combined_asps)

        combined[category_key] = combined_asps

        print(f"  {category_name}: {len(openai_sample)} OpenAI + {len(esco_sample)} ESCO = {len(combined_asps)} total")

    print(f"{'=' * 80}\n")
    return combined


# ==============================================================================
# PATTERN-BASED MESSAGE GENERATION
# ==============================================================================


def generate_messages_with_patterns(
    aspirations_by_category: Dict[str, List[str]], samples_per_pattern: Dict[str, int] = None
) -> List[Tuple[str, str, str, str, int]]:
    """
    Generate messages using linguistic patterns with aspirations.

    Args:
        aspirations_by_category: Dict of category_key -> list of aspirations
        samples_per_pattern: Dict of pattern_key -> number of samples (None = use recommended)

    Returns:
        List of (message, category, content_category, linguistic_pattern, commitment_level) tuples
    """
    print(f"\n{'=' * 80}")
    print("GENERATING MESSAGES WITH LINGUISTIC PATTERNS")
    print(f"{'=' * 80}")

    all_messages = []

    # Calculate samples per pattern if not provided
    if samples_per_pattern is None:
        # Flatten all aspirations
        all_aspirations = []
        for aspirations in aspirations_by_category.values():
            all_aspirations.extend(aspirations)

        total_aspirations = len(all_aspirations)
        samples_per_pattern = {pattern_key: get_recommended_sample_count(total_aspirations, pattern_key) for pattern_key in get_all_pattern_keys()}

    print("\nTarget distribution:")
    for pattern_key, count in samples_per_pattern.items():
        pattern_name = get_pattern_name(pattern_key)
        percentage = (count / sum(samples_per_pattern.values())) * 100
        print(f"  {pattern_name}: {count} ({percentage:.1f}%)")

    # Generate messages for each pattern
    for pattern_key in get_all_pattern_keys():
        pattern_name = get_pattern_name(pattern_key)
        pattern_templates = get_pattern_templates(pattern_key)
        commitment_level = get_pattern_commitment_level(pattern_key)
        target_count = samples_per_pattern.get(pattern_key, 0)

        print(f"\n{pattern_name}:")
        print(f"  Target: {target_count} messages")
        print(f"  Templates: {len(pattern_templates)}")
        print(f"  Commitment Level: {commitment_level}/5")

        pattern_messages = []

        # Distribute samples across content categories
        categories = list(aspirations_by_category.keys())
        samples_per_category = target_count // len(categories)

        for category_key in categories:
            aspirations = aspirations_by_category[category_key]

            # Generate messages for this category with this pattern
            for i in range(samples_per_category):
                # Randomly select template and aspiration
                template = random.choice(pattern_templates)
                aspiration = random.choice(aspirations)

                # Generate message
                message = template.replace("[OBJ]", aspiration)

                pattern_messages.append(
                    (
                        message,
                        "aspirational",  # Main category
                        category_key,  # Content category (WHAT)
                        pattern_key,  # Linguistic pattern (HOW)
                        commitment_level,  # Commitment level (1-5)
                    )
                )

        all_messages.extend(pattern_messages)
        print(f"  ✓ Generated {len(pattern_messages)} messages")

    print(f"\n{'=' * 80}")
    print(f"Total messages: {len(all_messages):,}")
    print(f"{'=' * 80}")

    return all_messages


# ==============================================================================
# STATISTICS AND VALIDATION
# ==============================================================================


def print_statistics(messages: List[Tuple[str, str, str, str, int]]):
    """Print generation statistics with pattern analysis."""
    print("\n" + "=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)

    df = pd.DataFrame(messages, columns=["message", "category", "content_category", "linguistic_pattern", "commitment_level"])

    # Overall stats
    print("\nOverall:")
    print(f"  Total messages: {len(df):,}")
    print(f"  Unique messages: {df['message'].nunique():,}")
    print(f"  Content categories: {df['content_category'].nunique()}")
    print(f"  Linguistic patterns: {df['linguistic_pattern'].nunique()}")

    # By content category
    print("\nBy content category (WHAT):")
    for category_key in df["content_category"].unique():
        category_name = get_category_name(category_key)
        count = len(df[df["content_category"] == category_key])
        percentage = (count / len(df)) * 100
        print(f"  {category_name}: {count:,} ({percentage:.1f}%)")

    # By linguistic pattern
    print("\nBy linguistic pattern (HOW):")
    for pattern_key in df["linguistic_pattern"].unique():
        pattern_name = get_pattern_name(pattern_key)
        count = len(df[df["linguistic_pattern"] == pattern_key])
        percentage = (count / len(df)) * 100
        target_percentage = PATTERN_DISTRIBUTION.get(pattern_key, 0) * 100
        commitment = df[df["linguistic_pattern"] == pattern_key]["commitment_level"].iloc[0]
        print(f"  {pattern_name}: {count:,} ({percentage:.1f}% | target: {target_percentage:.0f}%) [Commitment: {commitment}/5]")

    # By commitment level
    print("\nBy commitment level:")
    for level in sorted(df["commitment_level"].unique(), reverse=True):
        count = len(df[df["commitment_level"] == level])
        percentage = (count / len(df)) * 100
        patterns = df[df["commitment_level"] == level]["linguistic_pattern"].unique()
        pattern_names = [get_pattern_name(p) for p in patterns]
        print(f"  Level {level}/5: {count:,} ({percentage:.1f}%) - {', '.join(pattern_names)}")

    # Average commitment level
    avg_commitment = df["commitment_level"].mean()
    print(f"\nAverage commitment level: {avg_commitment:.2f}/5")

    # Message length stats
    print("\nMessage length statistics:")
    df["length"] = df["message"].str.len()
    print(f"  Min: {df['length'].min()} chars")
    print(f"  Max: {df['length'].max()} chars")
    print(f"  Mean: {df['length'].mean():.1f} chars")
    print(f"  Median: {df['length'].median():.1f} chars")

    # Pattern validation
    print("\nPattern coverage validation:")
    messages_list = df["message"].tolist()
    detected_patterns = validate_pattern_coverage(messages_list)
    for pattern_key, count in detected_patterns.items():
        pattern_name = get_pattern_name(pattern_key)
        percentage = (count / len(messages_list)) * 100
        print(f"  {pattern_name}: {count} detected ({percentage:.1f}%)")


def print_samples(messages: List[Tuple[str, str, str, str, int]], samples_per_pattern: int = 3):
    """Print sample messages for each pattern."""
    print("\n" + "=" * 80)
    print("SAMPLE MESSAGES BY LINGUISTIC PATTERN")
    print("=" * 80)

    df = pd.DataFrame(messages, columns=["message", "category", "content_category", "linguistic_pattern", "commitment_level"])

    for pattern_key in df["linguistic_pattern"].unique():
        pattern_name = get_pattern_name(pattern_key)
        pattern_samples = df[df["linguistic_pattern"] == pattern_key].head(samples_per_pattern)
        commitment_level = pattern_samples["commitment_level"].iloc[0]

        print(f"\n{pattern_name} (Commitment: {commitment_level}/5):")
        for i, row in pattern_samples.iterrows():
            content_category = get_category_name(row["content_category"])
            print(f'  [{content_category}] "{row["message"]}"')


# ==============================================================================
# SAVE FUNCTIONS
# ==============================================================================


def save_to_csv(messages: List[Tuple[str, str, str, str, int]], output_path: str):
    """
    Save messages to CSV with all category dimensions.

    Args:
        messages: List of (message, category, content_category, linguistic_pattern, commitment_level) tuples
        output_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = pd.DataFrame(messages, columns=["message", "category", "content_category", "linguistic_pattern", "commitment_level"])

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Saved {len(messages):,} messages to: {output_path}")
    print("  Columns: message, category, content_category, linguistic_pattern, commitment_level")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate aspirational training data with linguistic patterns")
    parser.add_argument("--output", type=str, default="training/data/aspirational_messages_with_patterns.csv", help="Output CSV file path")
    parser.add_argument(
        "--aspirations-per-category", type=int, default=50, help="Number of aspirations to generate per content category (default: 50)"
    )
    parser.add_argument("--total-messages", type=int, default=1000, help="Total number of messages to generate (default: 1000)")
    parser.add_argument("--temperature", type=float, default=0.8, help="OpenAI temperature (default: 0.8)")
    parser.add_argument("--samples-per-pattern", type=str, help="Custom distribution as JSON: '{\"explicit_desire\": 250, ...}'")
    parser.add_argument("--categories", nargs="+", choices=get_all_category_keys(), help="Specific content categories to use (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--use-esco", action="store_true", help="Include ESCO dataset (European occupations and skills)")
    parser.add_argument("--esco-weight", type=float, default=0.5, help="Proportion of ESCO data to use (0.0-1.0, default: 0.5 = 50%% ESCO)")
    parser.add_argument("--esco-occupation-limit", type=int, default=500, help="Max ESCO occupations to load (default: 500)")
    parser.add_argument("--esco-skill-limit", type=int, default=500, help="Max ESCO skills to load (default: 500)")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 80)
    print("ASPIRATIONAL DATA GENERATION WITH LINGUISTIC PATTERNS")
    print("=" * 80)

    # Show configuration
    print("\nContent Categories (WHAT):")
    categories = args.categories or get_all_category_keys()
    for key in categories:
        print(f"  {key}: {get_category_name(key)}")

    print("\nLinguistic Patterns (HOW):")
    for key in get_all_pattern_keys():
        print(f"  {key}: {get_pattern_name(key)}")

    print("\nConfiguration:")
    print(f"  Output: {args.output}")
    print(f"  Aspirations per content category: {args.aspirations_per_category}")
    print(f"  Total messages target: {args.total_messages}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Random seed: {args.seed}")
    print("\nESCO Integration:")
    print(f"  Use ESCO: {args.use_esco}")
    if args.use_esco:
        print(f"  ESCO weight: {args.esco_weight * 100:.0f}% ESCO, {(1 - args.esco_weight) * 100:.0f}% OpenAI")
        print(f"  ESCO occupation limit: {args.esco_occupation_limit}")
        print(f"  ESCO skill limit: {args.esco_skill_limit}")

    # Initialize OpenAI client
    client = get_openai()

    # Step 1: Generate aspirations for each content category
    print(f"\n{'=' * 80}")
    print("STEP 1: GENERATING ASPIRATIONS (Verb Phrases)")
    print(f"{'=' * 80}")

    aspirations_by_category = {}
    for category_key in categories:
        aspirations = generate_aspirational_objects(client, category_key, num_aspirations=args.aspirations_per_category, temperature=args.temperature)
        aspirations_by_category[category_key] = aspirations
        time.sleep(1)  # Rate limiting

    # Step 1.5: Combine with ESCO data if requested
    if args.use_esco:
        aspirations_by_category = combine_openai_and_esco_aspirations(
            aspirations_by_category,
            esco_occupation_limit=args.esco_occupation_limit,
            esco_skill_limit=args.esco_skill_limit,
            esco_weight=args.esco_weight,
        )

    # Step 2: Generate messages using linguistic patterns
    print(f"\n{'=' * 80}")
    print("STEP 2: APPLYING LINGUISTIC PATTERNS")
    print(f"{'=' * 80}")

    # Parse custom distribution if provided
    samples_per_pattern = None
    if args.samples_per_pattern:
        import json

        samples_per_pattern = json.loads(args.samples_per_pattern)
    else:
        # Calculate recommended distribution
        samples_per_pattern = {pattern_key: get_recommended_sample_count(args.total_messages, pattern_key) for pattern_key in get_all_pattern_keys()}

    messages = generate_messages_with_patterns(aspirations_by_category, samples_per_pattern)

    # Step 3: Print statistics and samples
    print_statistics(messages)
    print_samples(messages, samples_per_pattern=3)

    # Step 4: Save to CSV
    save_to_csv(messages, args.output)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated data")
    print("  2. Merge with other training data")
    print("  3. Train the intent classifier model")
    print("\nThe dataset includes 5 columns:")
    print("  - 'message': The generated message")
    print("  - 'category': Always 'aspirational'")
    print("  - 'content_category': WHAT the aspiration is about (dream_roles, salary, etc.)")
    print("  - 'linguistic_pattern': HOW it's expressed (explicit, future, intentional, etc.)")
    print("  - 'commitment_level': Psychological commitment score (1-5, where 5=highest)")


if __name__ == "__main__":
    main()
