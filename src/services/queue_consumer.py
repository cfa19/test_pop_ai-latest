"""
Queue Consumer — event-driven processing of memory_extraction_queue.

Primary mode: sleeps until woken by webhook (POST /api/extract).
Safety net: wakes every QUEUE_POLL_INTERVAL seconds to catch stuck/missed items.

Uses FOR UPDATE SKIP LOCKED for concurrent-safe processing.
Recovers stuck items after 10 minutes.
"""

import asyncio
import logging
from datetime import UTC, datetime

from src.config import RPCFunctions, Tables, get_supabase
from src.services import runner_orchestrator

logger = logging.getLogger(__name__)


class QueueConsumer:
    """Event-driven consumer with safety-net polling."""

    def __init__(
        self,
        poll_interval: float = 60.0,
        batch_size: int = 10,
        max_retries: int = 3,
    ):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._running = False
        self._wake_event = asyncio.Event()

    def wake(self):
        """Signal the consumer to process immediately (called by webhook)."""
        self._wake_event.set()

    async def start(self):
        """Start the consumer loop. Called from FastAPI lifespan."""
        self._running = True
        logger.info(f"QueueConsumer started (safety-poll={self.poll_interval}s, batch={self.batch_size}, retries={self.max_retries})")
        while self._running:
            try:
                processed = await self._poll_batch()
                if processed > 0:
                    continue  # More items may be waiting, check immediately
                # Wait for webhook wake OR safety-net timeout
                self._wake_event.clear()
                try:
                    await asyncio.wait_for(
                        self._wake_event.wait(),
                        timeout=self.poll_interval,
                    )
                    logger.debug("QueueConsumer woken by webhook")
                except asyncio.TimeoutError:
                    pass  # Safety-net poll
            except Exception as e:
                logger.error(f"Queue poll error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def stop(self):
        """Stop the consumer loop gracefully."""
        self._running = False
        self._wake_event.set()  # Unblock the wait
        logger.info("QueueConsumer stopped")

    async def _poll_batch(self) -> int:
        """Fetch and process a batch of pending items. Returns count processed."""
        try:
            supabase = get_supabase()
            items = supabase.rpc(
                RPCFunctions.CLAIM_EXTRACTION_BATCH,
                {
                    "batch_size": self.batch_size,
                    "max_retries": self.max_retries,
                },
            ).execute()
        except Exception as e:
            logger.error(f"Failed to claim batch: {e}")
            return 0

        if not items.data:
            return 0

        logger.info(f"Claimed {len(items.data)} queue items for processing")

        for item in items.data:
            now = datetime.now(UTC).isoformat()
            try:
                status = await runner_orchestrator.process_queue_item(item)
                supabase.table(Tables.MEMORY_EXTRACTION_QUEUE).update(
                    {
                        "status": status,
                        "completed_at": now,
                    }
                ).eq("id", item["id"]).execute()
                logger.info(f"Queue item {item['id']}: {status}")
            except Exception as e:
                logger.error(f"Processing failed for queue item {item['id']}: {e}")
                supabase.table(Tables.MEMORY_EXTRACTION_QUEUE).update(
                    {
                        "status": "failed",
                        "attempts": item.get("attempts", 0) + 1,
                        "last_error": str(e)[:500],
                    }
                ).eq("id", item["id"]).execute()

        return len(items.data)
