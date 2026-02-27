"""
Queue Consumer — polls memory_extraction_queue for pending items and processes them.

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
    """Polls memory_extraction_queue for pending items."""

    def __init__(
        self,
        poll_interval: float = 5.0,
        batch_size: int = 10,
        max_retries: int = 3,
    ):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._running = False

    async def start(self):
        """Start the polling loop. Called from FastAPI lifespan."""
        self._running = True
        logger.info(
            f"QueueConsumer started (poll={self.poll_interval}s, "
            f"batch={self.batch_size}, retries={self.max_retries})"
        )
        while self._running:
            try:
                processed = await self._poll_batch()
                if processed == 0:
                    await asyncio.sleep(self.poll_interval)
                # If we processed items, immediately check for more
            except Exception as e:
                logger.error(f"Queue poll error: {e}")
                await asyncio.sleep(self.poll_interval * 2)

    async def stop(self):
        """Stop the polling loop gracefully."""
        self._running = False
        logger.info("QueueConsumer stopped")

    async def _poll_batch(self) -> int:
        """Fetch and process a batch of pending items. Returns count processed."""
        try:
            supabase = get_supabase()
            items = supabase.rpc(RPCFunctions.CLAIM_EXTRACTION_BATCH, {
                "batch_size": self.batch_size,
                "max_retries": self.max_retries,
            }).execute()
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
                supabase.table(Tables.MEMORY_EXTRACTION_QUEUE).update({
                    "status": status,
                    "completed_at": now,
                }).eq("id", item["id"]).execute()
                logger.info(f"Queue item {item['id']}: {status}")
            except Exception as e:
                logger.error(f"Processing failed for queue item {item['id']}: {e}")
                supabase.table(Tables.MEMORY_EXTRACTION_QUEUE).update({
                    "status": "failed",
                    "attempts": item.get("attempts", 0) + 1,
                    "last_error": str(e)[:500],
                }).eq("id", item["id"]).execute()

        return len(items.data)
