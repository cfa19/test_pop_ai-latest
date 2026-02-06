"""
Test both OpenAI and DistilBERT intent classifiers

Verifies that both classifiers:
1. Return valid IntentClassification objects
2. Classify messages into the correct 7 categories
3. Provide reasonable confidence scores
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.distilbert_classifier import get_distilbert_classifier
from src.agents.langgraph_workflow import MessageCategory, _classify_with_openai
from src.config import CHAT_MODEL, INTENT_CLASSIFIER_MODEL_PATH, get_openai

# Test messages covering all 7 categories
TEST_MESSAGES = [
    {"message": "What is machine learning?", "expected": MessageCategory.RAG_QUERY, "description": "Factual question about ML"},
    {
        "message": "I have 5 years of Python experience and worked on REST APIs",
        "expected": MessageCategory.PROFESSIONAL,
        "description": "Professional skills and experience",
    },
    {
        "message": "I value work-life balance and believe in continuous learning",
        "expected": MessageCategory.PSYCHOLOGICAL,
        "description": "Values and beliefs",
    },
    {
        "message": "I learn best through hands-on projects and video tutorials",
        "expected": MessageCategory.LEARNING,
        "description": "Learning preferences",
    },
    {"message": "My mentor helped me navigate my career transition", "expected": MessageCategory.SOCIAL, "description": "Mentorship and networking"},
    {"message": "I'm feeling overwhelmed and burned out with work", "expected": MessageCategory.EMOTIONAL, "description": "Emotional wellbeing"},
    {
        "message": "I want to become a senior data scientist in 3 years",
        "expected": MessageCategory.ASPIRATIONAL,
        "description": "Career goals and aspirations",
    },
]


async def test_openai_classifier():
    """Test OpenAI LLM-based classifier"""
    print("\n" + "=" * 80)
    print("Testing OpenAI LLM Classifier")
    print("=" * 80)

    chat_client = get_openai()
    results = []

    for test_case in TEST_MESSAGES:
        message = test_case["message"]
        expected = test_case["expected"]

        print(f"\nTest: {test_case['description']}")
        print(f"Message: {message}")

        try:
            classification = await _classify_with_openai(message, chat_client, CHAT_MODEL)

            print(f"✓ Category: {classification.category.value}")
            print(f"  Confidence: {classification.confidence:.2%}")
            print(f"  Reasoning: {classification.reasoning[:80]}...")

            is_correct = classification.category == expected
            results.append(
                {
                    "test": test_case["description"],
                    "expected": expected.value,
                    "predicted": classification.category.value,
                    "confidence": classification.confidence,
                    "correct": is_correct,
                }
            )

            if not is_correct:
                print(f"  ⚠ Expected {expected.value}, got {classification.category.value}")

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({"test": test_case["description"], "expected": expected.value, "predicted": "ERROR", "confidence": 0.0, "correct": False})

    # Print summary
    print("\n" + "=" * 80)
    print("OpenAI Classifier Summary")
    print("=" * 80)
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0

    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.2%}")

    return results


async def test_distilbert_classifier():
    """Test DistilBERT fine-tuned model classifier"""
    print("\n" + "=" * 80)
    print("Testing DistilBERT Model Classifier")
    print("=" * 80)

    # Check if model exists
    if not os.path.exists(INTENT_CLASSIFIER_MODEL_PATH):
        print(f"⚠ Model not found at: {INTENT_CLASSIFIER_MODEL_PATH}")
        print("Skipping DistilBERT test.")
        return []

    try:
        classifier = get_distilbert_classifier(INTENT_CLASSIFIER_MODEL_PATH)
        results = []

        for test_case in TEST_MESSAGES:
            message = test_case["message"]
            expected = test_case["expected"]

            print(f"\nTest: {test_case['description']}")
            print(f"Message: {message}")

            try:
                classification = await classifier.classify(message)

                print(f"✓ Category: {classification.category.value}")
                print(f"  Confidence: {classification.confidence:.2%}")
                print(f"  Reasoning: {classification.reasoning}")

                is_correct = classification.category == expected
                results.append(
                    {
                        "test": test_case["description"],
                        "expected": expected.value,
                        "predicted": classification.category.value,
                        "confidence": classification.confidence,
                        "correct": is_correct,
                    }
                )

                if not is_correct:
                    print(f"  ⚠ Expected {expected.value}, got {classification.category.value}")

            except Exception as e:
                print(f"✗ Error: {str(e)}")
                results.append(
                    {"test": test_case["description"], "expected": expected.value, "predicted": "ERROR", "confidence": 0.0, "correct": False}
                )

        # Print summary
        print("\n" + "=" * 80)
        print("DistilBERT Classifier Summary")
        print("=" * 80)
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0

        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"Average Confidence: {avg_confidence:.2%}")

        return results

    except ImportError as e:
        print(f"⚠ Cannot import DistilBERT dependencies: {str(e)}")
        print("Install with: pip install transformers torch")
        return []


async def compare_classifiers():
    """Compare OpenAI and DistilBERT classifiers side-by-side"""
    print("\n" + "=" * 80)
    print("Classifier Comparison")
    print("=" * 80)

    openai_results = await test_openai_classifier()
    distilbert_results = await test_distilbert_classifier()

    if not distilbert_results:
        print("\n⚠ DistilBERT test skipped or failed.")
        return

    # Compare results
    print("\n" + "=" * 80)
    print("Side-by-Side Comparison")
    print("=" * 80)
    print(f"{'Test':<30} {'Expected':<15} {'OpenAI':<15} {'DistilBERT':<15} {'Match'}")
    print("-" * 80)

    for openai_r, distilbert_r in zip(openai_results, distilbert_results):
        test = openai_r["test"][:28]
        expected = openai_r["expected"]
        openai_pred = openai_r["predicted"]
        distilbert_pred = distilbert_r["predicted"]
        match = "✓" if openai_pred == distilbert_pred else "✗"

        print(f"{test:<30} {expected:<15} {openai_pred:<15} {distilbert_pred:<15} {match}")

    # Agreement rate
    agreements = sum(1 for o, d in zip(openai_results, distilbert_results) if o["predicted"] == d["predicted"])
    total = len(openai_results)
    agreement_rate = (agreements / total * 100) if total > 0 else 0

    print("\n" + "=" * 80)
    print(f"Agreement Rate: {agreements}/{total} ({agreement_rate:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test intent classifiers")
    parser.add_argument("--classifier", choices=["openai", "distilbert", "both"], default="both", help="Which classifier to test")

    args = parser.parse_args()

    if args.classifier == "openai":
        asyncio.run(test_openai_classifier())
    elif args.classifier == "distilbert":
        asyncio.run(test_distilbert_classifier())
    else:
        asyncio.run(compare_classifiers())
