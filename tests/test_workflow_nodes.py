"""
Test LangGraph Workflow with All 9 Category Nodes

Tests the complete workflow with messages for each of the 9 categories:
- RAG_QUERY
- PROFESSIONAL
- PSYCHOLOGICAL
- LEARNING
- SOCIAL
- EMOTIONAL
- ASPIRATIONAL
- CHITCHAT
- OFF_TOPIC
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.langgraph_workflow import MessageCategory, run_workflow
from src.config import CHAT_MODEL, EMBED_DIMENSIONS, EMBED_MODEL, get_client_by_provider, get_openai, get_supabase

# Test messages for each category
TEST_MESSAGES = [
    {
        "message": "What is machine learning and how is it used in data science?",
        "expected": MessageCategory.RAG_QUERY,
        "description": "Factual question requiring knowledge retrieval",
    },
    {
        "message": "I have 5 years of Python experience and have built REST APIs with FastAPI",
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
    {
        "message": "My mentor has been instrumental in helping me navigate my career",
        "expected": MessageCategory.SOCIAL,
        "description": "Mentorship and networking",
    },
    {
        "message": "I'm feeling completely burned out and exhausted with my current job",
        "expected": MessageCategory.EMOTIONAL,
        "description": "Emotional wellbeing and stress",
    },
    {
        "message": "I want to become a senior data scientist within the next 3 years",
        "expected": MessageCategory.ASPIRATIONAL,
        "description": "Career goals and aspirations",
    },
    {"message": "Hey! How are you doing today?", "expected": MessageCategory.CHITCHAT, "description": "Casual greeting"},
    {"message": "What's the weather like in New York?", "expected": MessageCategory.OFF_TOPIC, "description": "Unrelated to career"},
]


async def test_workflow():
    """Test the complete workflow with all 9 categories"""

    print("=" * 80)
    print("Testing LangGraph Workflow with All 9 Categories")
    print("=" * 80)

    # Initialize clients
    chat_client = get_openai()
    embed_client = get_client_by_provider("voyage")
    supabase = get_supabase()

    results = []

    for i, test_case in enumerate(TEST_MESSAGES, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}/{len(TEST_MESSAGES)}: {test_case['description']}")
        print(f"{'=' * 80}")
        print(f"Message: {test_case['message']}")
        print(f"Expected: {test_case['expected'].value}")

        try:
            # Run workflow
            final_state = await run_workflow(
                message=test_case["message"],
                user_id="test-user-123",
                conversation_id="test-conv-456",
                chat_client=chat_client,
                embed_client=embed_client,
                supabase=supabase,
                embed_model=EMBED_MODEL,
                embed_dimensions=EMBED_DIMENSIONS,
                chat_model=CHAT_MODEL,
            )

            # Extract results
            classification = final_state.get("unified_classification")
            response = final_state.get("response")
            metadata = final_state.get("metadata", {})

            if classification:
                print(f"\n✓ Classified as: {classification.category.value}")
                print(f"  Confidence: {classification.confidence:.2%}")
                print(f"  Reasoning: {classification.reasoning[:100]}...")

            if response:
                print(f"\n✓ Response generated ({len(response)} chars)")
                print(f"  Preview: {response[:150]}...")

            # Check if classification matches expected
            is_correct = classification and classification.category == test_case["expected"]

            results.append(
                {
                    "test": test_case["description"],
                    "expected": test_case["expected"].value,
                    "predicted": classification.category.value if classification else "NONE",
                    "confidence": classification.confidence if classification else 0.0,
                    "correct": is_correct,
                    "response_length": len(response) if response else 0,
                    "classifier_type": metadata.get("classifier_type", "unknown"),
                }
            )

            if not is_correct:
                print(f"\n  ⚠ Expected {test_case['expected'].value}, got {classification.category.value if classification else 'NONE'}")

        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            results.append(
                {
                    "test": test_case["description"],
                    "expected": test_case["expected"].value,
                    "predicted": "ERROR",
                    "confidence": 0.0,
                    "correct": False,
                    "response_length": 0,
                    "error": str(e),
                }
            )

        # Small delay between tests
        await asyncio.sleep(1)

    # Print summary
    print("\n" + "=" * 80)
    print("WORKFLOW TEST SUMMARY")
    print("=" * 80)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0
    avg_response_length = sum(r["response_length"] for r in results) / total if total > 0 else 0

    print("\nOverall Performance:")
    print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Average Confidence: {avg_confidence:.2%}")
    print(f"  Average Response Length: {avg_response_length:.0f} chars")

    print("\nDetailed Results:")
    print(f"{'Test':<40} {'Expected':<15} {'Predicted':<15} {'Correct'}")
    print("-" * 80)
    for r in results:
        test = r["test"][:38]
        expected = r["expected"]
        predicted = r["predicted"]
        correct_mark = "✓" if r["correct"] else "✗"
        print(f"{test:<40} {expected:<15} {predicted:<15} {correct_mark}")

    # Check for critical failures
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n⚠ {len(errors)} test(s) failed with errors:")
        for r in errors:
            print(f"  - {r['test']}: {r.get('error', 'Unknown error')}")

    return results


async def test_specific_category(category: MessageCategory):
    """Test workflow with a specific category"""

    # Find test case for category
    test_case = next((t for t in TEST_MESSAGES if t["expected"] == category), None)

    if not test_case:
        print(f"No test case found for category: {category.value}")
        return

    print(f"\n{'=' * 80}")
    print(f"Testing Specific Category: {category.value}")
    print(f"{'=' * 80}")

    # Initialize clients
    chat_client = get_openai()
    embed_client = get_client_by_provider("voyage")
    supabase = get_supabase()

    print(f"Message: {test_case['message']}")

    try:
        # Run workflow
        final_state = await run_workflow(
            message=test_case["message"],
            user_id="test-user-123",
            conversation_id="test-conv-456",
            chat_client=chat_client,
            embed_client=embed_client,
            supabase=supabase,
            embed_model=EMBED_MODEL,
            embed_dimensions=EMBED_DIMENSIONS,
            chat_model=CHAT_MODEL,
        )

        # Print full results
        classification = final_state.get("unified_classification")
        response = final_state.get("response")
        metadata = final_state.get("metadata", {})

        print("\nClassification:")
        print(f"  Category: {classification.category.value}")
        print(f"  Confidence: {classification.confidence:.2%}")
        print(f"  Reasoning: {classification.reasoning}")
        if classification.secondary_categories:
            print(f"  Secondary: {[c.value for c in classification.secondary_categories]}")
        if classification.key_entities:
            print(f"  Entities: {classification.key_entities}")

        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        print("\nResponse:")
        print(f"{response}")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LangGraph workflow nodes")
    parser.add_argument("--category", choices=[c.value for c in MessageCategory], help="Test a specific category")

    args = parser.parse_args()

    if args.category:
        # Test specific category
        category = MessageCategory(args.category)
        asyncio.run(test_specific_category(category))
    else:
        # Test all categories
        asyncio.run(test_workflow())
