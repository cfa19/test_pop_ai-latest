"""
Example: End-to-End Hierarchical Classification

This script demonstrates a complete workflow:
1. Load the hierarchical classifier
2. Classify various message types
3. Show detailed results with confidence scores
4. Demonstrate batch classification
"""

from test_hierarchical_classifier import HierarchicalClassifier


def format_result(result: dict) -> str:
    """Format a classification result for display."""
    lines = [
        f"Message: \"{result['message']}\"",
        f"  ├─ Category: {result['category']} ({result['category_confidence']:.1%})"
    ]

    if result['subcategory']:
        lines.append(
            f"  └─ Subcategory: {result['subcategory']} ({result['subcategory_confidence']:.1%})"
        )
    else:
        lines.append("  └─ Subcategory: None (no secondary classifier)")

    return "\n".join(lines)


def main():
    # Sample messages covering different categories
    test_messages = [
        # RAG queries
        "What is PopCoach?",
        "How much does the PopCoach subscription cost?",
        "what's the difference between PopSkills and PopCoach",

        # Professional context
        "I have 5 years of experience in Python development",
        "I'm skilled at project management and team leadership",
        "I recently earned my AWS certification",

        # Psychological context
        "I'm an introvert who prefers working independently",
        "I value work-life balance above all else",
        "I'm naturally detail-oriented and methodical",

        # Learning context
        "I learn best through hands-on practice",
        "I prefer video tutorials over reading documentation",
        "I completed my nursing degree last year",

        # Social context
        "I have a mentor who helps me with career decisions",
        "My colleagues say I'm a great team player",
        "I'm part of a local tech community",

        # Emotional context
        "I feel burned out from the constant deadlines",
        "I'm confident in my technical abilities",
        "I struggle with imposter syndrome sometimes",

        # Aspirational context
        "I want to become a data scientist",
        "My goal is to lead my own team within 2 years",
        "I dream of starting my own consulting business",

        # Chitchat
        "hello there",
        "thanks for your help",
        "see you later",

        # Off-topic
        "I love chocolate ice cream",
        "The weather is nice today",
        "What's your favorite movie?",
    ]

    # Load classifier (replace TIMESTAMP with your actual model directory)
    print("=" * 60)
    print("HIERARCHICAL CLASSIFIER EXAMPLE")
    print("=" * 60)
    print("\nNOTE: Update the model_dir path to your trained model!")
    print("Example: training/models/hierarchical/20260206_123456\n")

    model_dir = input("Enter model directory path: ").strip()

    print(f"\nLoading classifier from {model_dir}...")
    classifier = HierarchicalClassifier(model_dir)

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60 + "\n")

    # Classify all messages
    results = classifier.predict_batch(test_messages)

    # Group by category for organized display
    by_category = {}
    for result in results:
        category = result['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)

    # Display results grouped by category
    for category in sorted(by_category.keys()):
        category_results = by_category[category]
        print(f"\n{category.upper()} ({len(category_results)} messages)")
        print("-" * 60)

        for result in category_results:
            print(format_result(result))
            print()

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    category_counts = {}
    subcategory_counts = {}

    for result in results:
        # Count categories
        cat = result['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

        # Count subcategories
        if result['subcategory']:
            subcat = f"{cat}.{result['subcategory']}"
            subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

    print("\nCategory Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        percentage = count / len(results) * 100
        print(f"  {category:15s}: {count:2d} ({percentage:5.1f}%)")

    print(f"\nTotal messages: {len(results)}")
    print(f"Unique categories: {len(category_counts)}")
    print(f"Messages with subcategories: {len(subcategory_counts)}")

    # Average confidence scores
    avg_primary = sum(r['category_confidence'] for r in results) / len(results)
    secondary_confs = [r['subcategory_confidence'] for r in results if r['subcategory_confidence']]
    avg_secondary = sum(secondary_confs) / len(secondary_confs) if secondary_confs else 0

    print(f"\nAverage primary confidence: {avg_primary:.1%}")
    if avg_secondary > 0:
        print(f"Average secondary confidence: {avg_secondary:.1%}")

    # Most confident predictions
    print("\nTop 5 Most Confident Predictions:")
    sorted_by_conf = sorted(results, key=lambda x: x['category_confidence'], reverse=True)[:5]
    for i, result in enumerate(sorted_by_conf, 1):
        msg = result['message'][:50] + "..." if len(result['message']) > 50 else result['message']
        print(f"  {i}. {msg}")
        print(f"     {result['category']} ({result['category_confidence']:.1%})")

    # Least confident predictions (potential edge cases)
    print("\nTop 5 Least Confident Predictions (Review These):")
    sorted_by_conf_asc = sorted(results, key=lambda x: x['category_confidence'])[:5]
    for i, result in enumerate(sorted_by_conf_asc, 1):
        msg = result['message'][:50] + "..." if len(result['message']) > 50 else result['message']
        print(f"  {i}. {msg}")
        print(f"     {result['category']} ({result['category_confidence']:.1%})")

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
