"""
Idle background worker for span NER + Harmonia store.

Runs ``run_span_pipeline_background`` continuously when the server has no
active API requests.  As soon as an endpoint is called the worker pauses;
it resumes once all requests have finished.

Usage (in main.py):
    from src.utils.idle_worker import (
        start_idle_worker,
        stop_idle_worker,
        notify_request_start,
        notify_request_end,
    )

    @app.on_event("startup")
    async def startup():
        start_idle_worker()

    @app.on_event("shutdown")
    async def shutdown():
        await stop_idle_worker()

    @app.middleware("http")
    async def track_requests(request, call_next):
        notify_request_start()
        try:
            return await call_next(request)
        finally:
            notify_request_end()
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# Number of currently active HTTP requests.
_active_requests: int = 0

# Set by notify_request_start / cleared by notify_request_end.
# ``run_span_pipeline_background`` checks this to know when to pause.
_stop_event: asyncio.Event = asyncio.Event()

# Handle to the long-running worker task.
_worker_task: asyncio.Task | None = None

# Seconds to wait between full passes when no unprocessed messages are found.
_IDLE_POLL_INTERVAL: int = 30


# ---------------------------------------------------------------------------
# Request tracking helpers (called by middleware)
# ---------------------------------------------------------------------------

def notify_request_start() -> None:
    """Signal that an API request has started.  Pauses the idle worker."""
    global _active_requests
    _active_requests += 1
    if not _stop_event.is_set():
        _stop_event.set()
        logger.debug("[IDLE] Request started — background worker paused")


def notify_request_end() -> None:
    """Signal that an API request has finished.  Resumes the idle worker if
    no other requests are in flight."""
    global _active_requests
    _active_requests = max(0, _active_requests - 1)
    if _active_requests == 0 and _stop_event.is_set():
        _stop_event.clear()
        logger.debug("[IDLE] All requests done — background worker resumed")


# ---------------------------------------------------------------------------
# Idle worker loop
# ---------------------------------------------------------------------------

async def _idle_worker_loop() -> None:
    """Main loop: wait until idle, then process one full pass of unprocessed
    messages.  Exits cleanly on CancelledError (server shutdown)."""
    from src.agents.langgraph_workflow import run_span_pipeline_background
    from src.config import (
        CHAT_MODEL,
        EMBED_DIMENSIONS,
        EMBED_MODEL,
        SEMANTIC_GATE_ENABLED,
        get_client_by_provider,
        get_openai,
        get_supabase,
    )

    logger.info("[IDLE] Idle background worker started")

    embed_provider = "voyage" if EMBED_MODEL.startswith("voyage") else "openai"

    while True:
        try:
            # ── Wait until no active requests ─────────────────────────────
            while _stop_event.is_set():
                await asyncio.sleep(0.2)

            # ── Run one full pass across all unprocessed messages ─────────
            logger.debug("[IDLE] Running background span pipeline pass...")

            supabase     = get_supabase()
            chat_client  = get_openai()
            embed_client = get_client_by_provider(embed_provider)

            await run_span_pipeline_background(
                # No "current message" — process everything from the DB.
                current_message="",
                current_all_spans=[],
                # user_id / conversation_id = None → query all conversations.
                user_id=None,
                conversation_id=None,
                supabase=supabase,
                chat_client=chat_client,
                embed_client=embed_client,
                embed_model=EMBED_MODEL,
                embed_dimensions=EMBED_DIMENSIONS,
                chat_model=CHAT_MODEL,
                auth_header=None,
                semantic_gate_enabled=SEMANTIC_GATE_ENABLED,
                stop_event=_stop_event,
            )

            # ── Brief pause before the next pass ──────────────────────────
            await asyncio.sleep(_IDLE_POLL_INTERVAL)

        except asyncio.CancelledError:
            logger.info("[IDLE] Idle worker cancelled — shutting down")
            raise
        except Exception as exc:
            logger.warning(f"[IDLE] Error in idle worker loop: {exc}")
            await asyncio.sleep(5)  # back-off before retrying


# ---------------------------------------------------------------------------
# Lifecycle helpers (called from main.py)
# ---------------------------------------------------------------------------

def start_idle_worker() -> None:
    """Create and schedule the idle worker task.  Must be called after the
    asyncio event loop is running (e.g., inside a FastAPI startup handler)."""
    global _worker_task
    if _worker_task is not None and not _worker_task.done():
        logger.warning("[IDLE] Worker already running — ignoring start request")
        return
    _worker_task = asyncio.create_task(_idle_worker_loop(), name="idle-span-worker")
    logger.info("[IDLE] Idle worker task created")


async def stop_idle_worker() -> None:
    """Cancel the idle worker task and wait for it to finish."""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    _worker_task = None
    logger.info("[IDLE] Idle worker stopped")
