"""
Test script for ONNX classifier.

Usage:
    1. First export the model:
       python training/scripts/export_onnx.py --quantize

    2. Then run this test:
       python test_onnx.py
"""

import asyncio
import time


async def test_onnx_classifier():
    """Test the ONNX classifier with sample messages."""

    from src.agents.onnx_classifier import get_onnx_classifier

    # Test messages
    test_messages = [
        ("Hola, como estas?", "chitchat"),
        ("Quiero ser CTO en 5 años", "aspirational"),
        ("Tengo 10 años de experiencia en Python", "professional"),
        ("Me siento frustrado con mi trabajo", "emotional"),
        ("Que es POP Skills?", "rag_query"),
        ("Como puedo mejorar mis habilidades de liderazgo?", "learning"),
        ("Receta de pizza", "off_topic"),
    ]

    print("Loading ONNX classifier...")

    # Initialize classifier
    try:
        classifier = get_onnx_classifier(model_path="training/models/onnx/classifier")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("1. Run: python training/scripts/export_onnx.py --quantize")
        print("2. Copy centroids from pop-ai project")
        return

    print("\nTesting classification:\n")
    print("-" * 70)

    total_time = 0
    correct = 0

    for message, expected in test_messages:
        start = time.perf_counter()
        result = await classifier.classify(message)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        is_correct = result.category.value == expected
        if is_correct:
            correct += 1

        status = "OK" if is_correct else "X "
        print(f"{status} '{message[:40]:<40}' -> {result.category.value:<15} (conf: {result.confidence:.2f}, {elapsed:.1f}ms)")

    print("-" * 70)
    print(f"\nAccuracy: {correct}/{len(test_messages)} ({100 * correct / len(test_messages):.0f}%)")
    print(f"Avg latency: {total_time / len(test_messages):.1f}ms")
    print(f"Total time: {total_time:.1f}ms")


if __name__ == "__main__":
    asyncio.run(test_onnx_classifier())
