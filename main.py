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
from src.utils.message_queue import start_message_queue, stop_message_queue

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===============================
# FASTAPI APP
# ===============================
app = FastAPI(title="Pop Skills Chatbot API", description="API for the chatbot with RAG", version="1.0.0")


def download_models():
    """Download ONNX models from HuggingFace Hub if not available locally."""
    from pathlib import Path

    from src.config import HF_REPO_ID, INTENT_CLASSIFIER_MODEL_PATH, ONNX_HIERARCHY_PATH

    # Check if models already exist locally (e.g., dev environment)
    primary_exists = Path(INTENT_CLASSIFIER_MODEL_PATH).exists()
    hierarchy_exists = (Path(ONNX_HIERARCHY_PATH) / "secondary").exists()

    if primary_exists and hierarchy_exists:
        logger.info("ONNX models found locally, skipping HF Hub download")
        return

    logger.info(f"Downloading ONNX models from HuggingFace Hub: {HF_REPO_ID}")
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(repo_id=HF_REPO_ID)

    # Update config paths to point to downloaded models
    config.INTENT_CLASSIFIER_MODEL_PATH = str(Path(local_path) / "classifier")
    config.ONNX_HIERARCHY_PATH = local_path
    config.SEMANTIC_GATE_ONNX_MODEL_PATH = str(Path(local_path) / "semantic_gate")
    config.SEMANTIC_GATE_CENTROIDS_DIR = str(Path(local_path) / "semantic_gate")
    config.SEMANTIC_GATE_TUNING_PATH = str(Path(local_path) / "semantic_gate" / "semantic_gate_hierarchical_tuning.json")

    logger.info(f"Models downloaded to {local_path}")
    logger.info(f"  Primary classifier: {config.INTENT_CLASSIFIER_MODEL_PATH}")
    logger.info(f"  ONNX hierarchy: {config.ONNX_HIERARCHY_PATH}")
    logger.info(f"  Semantic gate ONNX: {config.SEMANTIC_GATE_ONNX_MODEL_PATH}")


async def preload_models():
    """Preload ML models on startup."""
    from src.config import (
        INTENT_CLASSIFIER_MODEL_PATH,
        PRIMARY_INTENT_CLASSIFIER_TYPE,
        SEMANTIC_GATE_ENABLED,
        SEMANTIC_GATE_MODEL,
        set_intent_classifier,
        set_semantic_gate,
    )

    # Preload intent classifier
    if PRIMARY_INTENT_CLASSIFIER_TYPE == "onnx":
        try:
            logger.info("Preloading ONNX intent classifier...")
            from src.agents.onnx_classifier import get_onnx_classifier

            classifier = get_onnx_classifier(INTENT_CLASSIFIER_MODEL_PATH)
            set_intent_classifier(classifier)

            logger.info("ONNX intent classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload ONNX classifier: {e}")
            logger.warning("Will fall back to lazy loading on first request")
    else:
        logger.info(f"Intent classifier type: {PRIMARY_INTENT_CLASSIFIER_TYPE} (no preloading needed)")

    # Preload semantic gate if enabled
    if SEMANTIC_GATE_ENABLED:
        try:
            logger.info("Preloading semantic gate (ONNX)...")
            from src.agents.semantic_gate_onnx import get_semantic_gate_onnx

            gate = get_semantic_gate_onnx()
            set_semantic_gate(gate)

            logger.info("Semantic gate (ONNX) loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload ONNX semantic gate: {e}")
            try:
                logger.info("Trying PyTorch semantic gate as fallback...")
                from src.agents.semantic_gate import get_semantic_gate

                gate = get_semantic_gate(model_name=SEMANTIC_GATE_MODEL)
                set_semantic_gate(gate)
                logger.info("PyTorch semantic gate loaded successfully")
            except Exception as e2:
                logger.warning(f"Failed to preload semantic gate: {e2}")
                logger.warning("Will fall back to lazy loading on first request")
    else:
        logger.info("Semantic gate disabled (no preloading needed)")


async def _background_model_setup():
    """Download and preload models in background so the server port opens immediately."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, download_models)
        await preload_models()
        config.MODELS_READY = True
        logger.info("Background model setup complete - ready to serve requests")
    except Exception as e:
        logger.error(f"Background model setup failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("Starting Pop Skills AI API...")

    # Start message queue for sequential processing (fast, non-blocking)
    await start_message_queue()
    logger.info("Message queue initialized")

    # Download and preload models in background (don't block port binding)
    asyncio.create_task(_background_model_setup())
    logger.info("Model download/preload scheduled in background")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logger.info("Shutting down Pop Skills AI API...")

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
