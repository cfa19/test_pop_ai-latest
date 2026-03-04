"""
Backend FastAPI for the Chatbot with RAG
"""

import argparse
import asyncio
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import src.config as config
from src.api.chat import router as chat_router
from src.api.webhooks import router as webhooks_router
from src.services.queue_consumer import QueueConsumer
from src.utils.idle_worker import (
    notify_request_end,
    notify_request_start,
    stop_idle_worker,
)
from src.utils.message_queue import start_message_queue, stop_message_queue

# Module-level reference for graceful shutdown
_queue_consumer: QueueConsumer | None = None


def get_queue_consumer() -> QueueConsumer | None:
    """Return the running QueueConsumer instance (used by webhooks)."""
    return _queue_consumer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# FASTAPI APP
# ===============================
app = FastAPI(
    title="Pop Skills Chatbot API",
    description="API for the chatbot with RAG",
    version="1.0.0"
)


def download_models_from_hf(local_dir: str):
    """Download ONNX models from HuggingFace Hub if not present locally."""
    from pathlib import Path

    hf_repo = os.getenv("HF_REPO_ID")
    if not hf_repo:
        logger.info("HF_REPO_ID not set, skipping HuggingFace download")
        return

    local_path = Path(local_dir)
    # Skip if models already exist locally
    if local_path.exists() and any(local_path.rglob("*.onnx")):
        logger.info(f"ONNX models already present at {local_dir}, skipping download")
        return

    logger.info(f"Downloading ONNX models from huggingface.co/{hf_repo} → {local_dir}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=hf_repo,
            local_dir=local_dir,
            repo_type="model",
        )
        logger.info(f"Downloaded ONNX models to {local_dir}")
    except Exception as e:
        logger.warning(f"Failed to download models from HF Hub: {e}")


async def preload_models():
    """Preload ML models on startup."""
    from src.config import (
        ONNX_MODELS_PATH,
        SEMANTIC_GATE_ENABLED,
        SEMANTIC_GATE_MODEL,
    )

    # Download from HuggingFace Hub if needed (deploy scenario)
    download_models_from_hf(ONNX_MODELS_PATH)

    # Preload semantic gate if enabled
    if SEMANTIC_GATE_ENABLED:
        try:
            logger.info("Preloading semantic gate...")
            from pathlib import Path

            from src.agents.semantic_gate import get_semantic_gate

            centroids_dir = Path(ONNX_MODELS_PATH) / "semantic_gate"
            gate = get_semantic_gate(
                centroids_dir=str(centroids_dir),
                model_name=SEMANTIC_GATE_MODEL,
            )

            if gate.is_hierarchical:
                logger.info("Semantic gate loaded (HIERARCHICAL mode)")
                logger.info(f"  Primary categories: {len(gate.primary_thresholds)}")
                total_secondary = sum(len(subcats) for subcats in gate.secondary_thresholds.values())
                logger.info(f"  Secondary categories: {total_secondary}")
            else:
                logger.info("Semantic gate loaded (legacy mode)")
                logger.info(f"  Categories: {len(gate.primary_thresholds)}")
        except Exception as e:
            logger.warning(f"Failed to preload semantic gate: {e}")
            logger.warning("Will fall back to lazy loading on first request")
    else:
        logger.info("Semantic gate disabled (no preloading needed)")


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting Pop Skills AI API...")

    # Start message queue for sequential processing
    await start_message_queue()
    logger.info("Message queue initialized")

    # Preload ML models (intent classifier, semantic gate)
    await preload_models()
    logger.info("Model preloading complete")

    # # Start idle background worker (NER + Harmonia store for context spans)
    # start_idle_worker()
    # logger.info("Idle span-pipeline worker started")

    # Start runner extraction queue consumer
    if config.RUNNER_EXTRACTION_ENABLED:
        global _queue_consumer
        _queue_consumer = QueueConsumer(
            poll_interval=config.QUEUE_POLL_INTERVAL,
            batch_size=config.QUEUE_BATCH_SIZE,
            max_retries=config.QUEUE_MAX_RETRIES,
        )
        asyncio.create_task(_queue_consumer.start())
        logger.info("Runner extraction queue consumer started")
    else:
        logger.info("Runner extraction disabled (RUNNER_EXTRACTION_ENABLED=false)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logger.info("Shutting down Pop Skills AI API...")

    # Stop runner extraction queue consumer
    global _queue_consumer
    if _queue_consumer:
        await _queue_consumer.stop()
        _queue_consumer = None
        logger.info("Runner extraction queue consumer stopped")

    # Stop idle background worker
    await stop_idle_worker()
    logger.info("Idle span-pipeline worker stopped")

    # Stop message queue
    await stop_message_queue()
    logger.info("Message queue stopped")


@app.middleware("http")
async def track_active_requests(request: Request, call_next):
    """Pause the idle span-pipeline worker while a request is being served."""
    notify_request_start()
    try:
        return await call_next(request)
    finally:
        notify_request_end()

# Allow CORS (for the frontend to call the backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat_router)
app.include_router(webhooks_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pop Skills AI Chatbot API")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode to include workflow process information in responses")
    args = parser.parse_args()

    # Set verbose mode in config
    if args.verbose:
        config.VERBOSE_MODE = True
        print("[INFO] Verbose mode enabled - workflow process will be included in responses")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
