"""
Backend FastAPI for the Chatbot with RAG
"""

import argparse
import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.config as config
from src.api.chat import router as chat_router
from src.services.queue_consumer import QueueConsumer
from src.utils.message_queue import start_message_queue, stop_message_queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Queue consumer instance (set during startup)
_queue_consumer: QueueConsumer | None = None

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
    """Preload ONNX token classifiers on startup."""
    from src.config import ONNX_MODELS_PATH, set_hierarchical_classifier

    # Download from HuggingFace Hub if needed (deploy scenario)
    download_models_from_hf(ONNX_MODELS_PATH)

    try:
        from src.agents.onnx_classifier import HierarchicalONNXTokenClassifier

        logger.info(f"Preloading ONNX token classifiers from {ONNX_MODELS_PATH}...")
        clf = HierarchicalONNXTokenClassifier(ONNX_MODELS_PATH)
        set_hierarchical_classifier(clf)
        logger.info(
            f"Loaded: primary + {len(clf.secondary)} secondary classifiers "
            f"({', '.join(clf.secondary.keys())})"
        )
    except Exception as e:
        logger.warning(f"Failed to preload ONNX classifiers: {e}")
        logger.warning("Will fall back to lazy loading on first request")


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


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logger.info("Shutting down Pop Skills AI API...")

    # Stop runner extraction queue consumer
    if _queue_consumer:
        await _queue_consumer.stop()
        logger.info("Runner extraction queue consumer stopped")

    # Stop message queue
    await stop_message_queue()
    logger.info("Message queue stopped")

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
