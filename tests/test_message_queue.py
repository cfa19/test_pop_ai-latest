"""
Test script for message queue system.

Simulates concurrent requests to verify sequential processing.
"""

import asyncio
import time

from src.utils.message_queue import MessageQueue


async def mock_workflow(message: str, user_id: str, delay: float = 0.5) -> dict:
    """
    Mock workflow function that simulates processing time.

    Args:
        message: User message
        user_id: User ID
        delay: Simulated processing time

    Returns:
        Mock workflow state
    """
    print(f"  [WORKER] Processing message from {user_id}: '{message[:30]}...'")
    await asyncio.sleep(delay)  # Simulate model inference
    print(f"  [WORKER] Completed message from {user_id}")

    return {
        "response": f"Response to: {message}",
        "user_id": user_id,
        "metadata": {"processed": True}
    }


async def test_sequential_processing():
    """Test that messages are processed sequentially."""
    print("\n" + "="*80)
    print("TEST 1: Sequential Processing")
    print("="*80)

    queue = MessageQueue()
    await queue.start()

    # Submit 5 messages concurrently
    messages = [
        ("Hello, how are you?", "user1"),
        ("I want to become a software engineer", "user2"),
        ("Me siento estresado", "user3"),
        ("What is machine learning?", "user4"),
        ("I have 5 years of Python experience", "user5"),
    ]

    print(f"\n[TEST] Submitting {len(messages)} messages concurrently...")
    start_time = time.time()

    # Submit all at once (concurrent)
    tasks = [
        queue.process_message(
            workflow_func=mock_workflow,
            message=msg,
            user_id=user_id,
            delay=0.5  # Each takes 0.5s
        )
        for msg, user_id in messages
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    print("\n[TEST] All messages processed")
    print(f"[TEST] Total time: {elapsed:.2f}s")
    print(f"[TEST] Expected time (sequential): {len(messages) * 0.5:.2f}s")
    print("[TEST] Expected time (concurrent): 0.5s")

    # Verify sequential processing
    if elapsed >= (len(messages) * 0.5) - 0.2:  # Allow 0.2s tolerance
        print("[TEST] ✅ PASS: Messages processed sequentially")
    else:
        print("[TEST] ❌ FAIL: Messages may have been processed concurrently")

    # Verify all results
    assert len(results) == len(messages)
    for i, result in enumerate(results):
        assert result["user_id"] == messages[i][1]
        assert "Response to:" in result["response"]

    print(f"[TEST] ✅ All {len(results)} results received correctly")

    await queue.stop()
    print()


async def test_queue_size():
    """Test queue size tracking."""
    print("\n" + "="*80)
    print("TEST 2: Queue Size Tracking")
    print("="*80)

    queue = MessageQueue()
    await queue.start()

    async def slow_workflow(delay: float = 2.0) -> dict:
        """Slow workflow to test queue buildup."""
        await asyncio.sleep(delay)
        return {"done": True}

    print("\n[TEST] Submitting 3 slow messages (2s each)...")

    # Submit 3 messages quickly
    task1 = asyncio.create_task(
        queue.process_message(workflow_func=slow_workflow, delay=2.0)
    )

    await asyncio.sleep(0.1)  # Small delay
    task2 = asyncio.create_task(
        queue.process_message(workflow_func=slow_workflow, delay=2.0)
    )

    await asyncio.sleep(0.1)
    task3 = asyncio.create_task(
        queue.process_message(workflow_func=slow_workflow, delay=2.0)
    )

    # Check queue size
    await asyncio.sleep(0.2)  # Let them queue up
    queue_size = queue.get_queue_size()
    pending_count = queue.get_pending_count()

    print(f"[TEST] Queue size: {queue_size}")
    print(f"[TEST] Pending requests: {pending_count}")

    # Queue size should be 2 (one being processed, two waiting)
    if queue_size >= 1:
        print("[TEST] ✅ PASS: Queue is building up correctly")
    else:
        print("[TEST] ❌ FAIL: Queue is not tracking correctly")

    # Wait for all to finish
    await asyncio.gather(task1, task2, task3)

    # Queue should be empty now
    final_queue_size = queue.get_queue_size()
    print(f"[TEST] Final queue size: {final_queue_size}")

    if final_queue_size == 0:
        print("[TEST] ✅ PASS: Queue emptied after processing")
    else:
        print("[TEST] ❌ FAIL: Queue not empty after processing")

    await queue.stop()
    print()


async def test_error_handling():
    """Test error handling in queue."""
    print("\n" + "="*80)
    print("TEST 3: Error Handling")
    print("="*80)

    queue = MessageQueue()
    await queue.start()

    async def failing_workflow() -> dict:
        """Workflow that raises an error."""
        await asyncio.sleep(0.1)
        raise ValueError("Simulated error")

    print("\n[TEST] Submitting message that will fail...")

    try:
        result = await queue.process_message(workflow_func=failing_workflow)
        print("[TEST] ❌ FAIL: Expected exception was not raised")
    except ValueError as e:
        print(f"[TEST] ✅ PASS: Exception caught correctly: {e}")

    # Queue should still work after error
    print("\n[TEST] Submitting successful message after error...")

    async def success_workflow() -> dict:
        return {"success": True}

    result = await queue.process_message(workflow_func=success_workflow)

    if result["success"]:
        print("[TEST] ✅ PASS: Queue works correctly after error")
    else:
        print("[TEST] ❌ FAIL: Queue not working after error")

    await queue.stop()
    print()


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MESSAGE QUEUE SYSTEM TESTS")
    print("="*80)

    await test_sequential_processing()
    await test_queue_size()
    await test_error_handling()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
