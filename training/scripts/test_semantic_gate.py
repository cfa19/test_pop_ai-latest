"""
Test Semantic Gate Integration

Simple script to test the semantic gate with sample messages.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.agents.semantic_gate import get_semantic_gate


def test_semantic_gate():
    """Test semantic gate with sample messages"""

    print("=" * 80)
    print("SEMANTIC GATE TEST")
    print("=" * 80)

    # Load semantic gate
    print("\n[1/3] Loading semantic gate...")
    try:
        gate = get_semantic_gate()
        print("✅ Semantic gate loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load semantic gate: {e}")
        return

    # Print statistics
    print("\n[2/3] Semantic gate statistics:")
    stats = gate.get_statistics()
    print(f"  Model: {stats['model_name']}")
    print(f"  Categories: {stats['num_categories']}")
    print(f"  Mean domain acceptance: {stats['global_metrics']['mean_domain_acceptance'] * 100:.2f}%")
    print(f"  Mean off-topic rejection: {stats['global_metrics']['mean_offtopic_rejection'] * 100:.2f}%")

    print("\n  Per-category thresholds:")
    for category, threshold in sorted(stats["thresholds"].items()):
        print(f"    {category:<20} {threshold:.4f}")

    # Test messages
    print("\n[3/3] Testing sample messages...")
    test_messages = [
        ("How do I become a data scientist?", "rag_query"),
        ("I have 5 years of Python experience", "professional"),
        ("I feel stressed about my job search", "emotional"),
        ("I want to become a CTO in 5 years", "aspirational"),
        ("What's the weather like today?", "chitchat"),
        ("I love pizza!", "off_topic"),
        ("Tell me about yourself", "chitchat"),
        ("I'm passionate about helping others", "psychological"),
        ("My mentor has been really helpful", "social"),
        ("I learn best through hands-on projects", "learning"),
    ]

    print(f"\n{'Message':<50} {'Category':<15} {'Pass?':<8} {'Similarity':<12} {'Threshold':<12}")
    print("-" * 100)

    for message, predicted_category in test_messages:
        should_pass, similarity, best_category = gate.check_message(message, predicted_category)
        threshold = gate.get_threshold(predicted_category)

        pass_str = "✅ PASS" if should_pass else "❌ BLOCK"
        message_short = message[:47] + "..." if len(message) > 47 else message

        print(f"{message_short:<50} {predicted_category:<15} {pass_str:<8} {similarity:<12.4f} {threshold:<12.4f}")

    print("\n" + "=" * 80)
    print("✅ Semantic gate test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_semantic_gate()
