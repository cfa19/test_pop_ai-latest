"""
Webhook endpoints for Supabase database triggers.

Supabase calls these via pg_net when rows are inserted into
monitored tables. This replaces constant polling with event-driven
processing.
"""

import logging

from fastapi import APIRouter, Response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/extract", status_code=204)
async def trigger_extraction():
    """Wake the queue consumer to process new extraction items.

    Called by Supabase pg_net trigger on INSERT into
    memory_extraction_queue. Returns 204 immediately — actual
    processing happens asynchronously in the consumer loop.
    """
    from main import get_queue_consumer

    consumer = get_queue_consumer()
    if consumer:
        consumer.wake()
    else:
        logger.warning("Extraction webhook called but queue consumer is not running")

    return Response(status_code=204)
