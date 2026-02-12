"""
Message Queue for Sequential Processing

Ensures messages are processed one at a time to avoid overwhelming
the GPU/CPU with concurrent classifier/model calls.

This is especially important for:
- DistilBERT intent classifier (GPU memory)
- Semantic gate (sentence transformer)
- Any other ML models that shouldn't run concurrently
"""

import asyncio
import logging
import uuid
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MessageQueue:
    """
    Async message queue for sequential workflow processing.

    Ensures only one message is processed at a time to avoid:
    - GPU memory overflow from concurrent model inference
    - CPU overload from multiple embeddings
    - Race conditions in model loading
    """

    def __init__(self):
        """Initialize the message queue."""
        self.queue: asyncio.Queue = asyncio.Queue()
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.worker_task: asyncio.Task | None = None
        self.is_running = False

    async def start(self):
        """Start the queue worker."""
        if self.is_running:
            logger.warning("[MESSAGE QUEUE] Already running")
            return

        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("[MESSAGE QUEUE] Started worker")

    async def stop(self):
        """Stop the queue worker gracefully."""
        if not self.is_running:
            return

        logger.info("[MESSAGE QUEUE] Stopping worker...")
        self.is_running = False

        # Cancel the worker task
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        # Reject all pending requests
        for request_id, future in self.pending_requests.items():
            if not future.done():
                future.set_exception(
                    RuntimeError("Message queue stopped before processing completed")
                )

        self.pending_requests.clear()
        logger.info("[MESSAGE QUEUE] Stopped")

    async def process_message(
        self,
        workflow_func,
        **kwargs
    ) -> Any:
        """
        Submit a message for processing and wait for the result.

        Args:
            workflow_func: Async function to execute (e.g., run_workflow)
            **kwargs: Arguments to pass to workflow_func

        Returns:
            Result from workflow_func

        Raises:
            Exception: If processing fails
        """
        if not self.is_running:
            raise RuntimeError("Message queue is not running. Call start() first.")

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Create a future to wait for the result
        result_future = asyncio.Future()
        self.pending_requests[request_id] = result_future

        # Add message to queue
        await self.queue.put({
            "request_id": request_id,
            "workflow_func": workflow_func,
            "kwargs": kwargs
        })

        logger.debug(f"[MESSAGE QUEUE] Queued request {request_id[:8]}... (queue size: {self.queue.qsize()})")

        # Wait for the result
        try:
            result = await result_future
            return result
        finally:
            # Clean up
            self.pending_requests.pop(request_id, None)

    async def _worker(self):
        """
        Worker task that processes messages sequentially.

        This ensures only one message is processed at a time,
        preventing concurrent GPU/CPU usage.
        """
        logger.info("[MESSAGE QUEUE] Worker started")

        while self.is_running:
            try:
                # Get next message from queue (wait up to 1 second)
                try:
                    message_data = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No message in queue, continue loop
                    continue

                request_id = message_data["request_id"]
                workflow_func = message_data["workflow_func"]
                kwargs = message_data["kwargs"]

                logger.debug(
                    f"[MESSAGE QUEUE] Processing request {request_id[:8]}... "
                    f"(queue size: {self.queue.qsize()})"
                )

                # Get the future for this request
                result_future = self.pending_requests.get(request_id)
                if not result_future or result_future.done():
                    # Request was cancelled or already completed
                    logger.warning(f"[MESSAGE QUEUE] Request {request_id[:8]}... was cancelled")
                    continue

                # Process the message (sequentially)
                try:
                    result = await workflow_func(**kwargs)
                    result_future.set_result(result)
                    logger.debug(f"[MESSAGE QUEUE] Completed request {request_id[:8]}...")

                except Exception as e:
                    logger.exception(f"[MESSAGE QUEUE] Error processing request {request_id[:8]}...")
                    result_future.set_exception(e)

            except asyncio.CancelledError:
                logger.info("[MESSAGE QUEUE] Worker cancelled")
                break
            except Exception as e:
                logger.exception(f"[MESSAGE QUEUE] Unexpected error in worker: {e}")
                # Continue processing other messages

        logger.info("[MESSAGE QUEUE] Worker stopped")

    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self.queue.qsize()

    def get_pending_count(self) -> int:
        """Get the number of pending requests."""
        return len(self.pending_requests)


# Global message queue instance
_message_queue: MessageQueue | None = None


def get_message_queue() -> MessageQueue:
    """
    Get or create the global message queue instance.

    Returns:
        MessageQueue instance
    """
    global _message_queue

    if _message_queue is None:
        _message_queue = MessageQueue()

    return _message_queue


async def start_message_queue():
    """Start the global message queue."""
    queue = get_message_queue()
    await queue.start()
    logger.info("[MESSAGE QUEUE] Global queue started")


async def stop_message_queue():
    """Stop the global message queue."""
    global _message_queue

    if _message_queue:
        await _message_queue.stop()
        _message_queue = None
        logger.info("[MESSAGE QUEUE] Global queue stopped")
