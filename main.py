"""
Backend FastAPI for the Chatbot with RAG
"""

import argparse
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.config as config
from src.api.chat import router as chat_router
from src.utils.message_queue import start_message_queue, stop_message_queue

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


async def preload_models():
    """Preload ML models on startup."""
    from src.config import (
        INTENT_CLASSIFIER_TYPE,
        INTENT_CLASSIFIER_MODEL_PATH,
        SEMANTIC_GATE_ENABLED,
        set_intent_classifier,
        set_semantic_gate
    )

    # Preload intent classifier
    if INTENT_CLASSIFIER_TYPE == "onnx":
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
        logger.info(f"Intent classifier type: {INTENT_CLASSIFIER_TYPE} (no preloading needed)")

    # Preload semantic gate if enabled
    if SEMANTIC_GATE_ENABLED:
        try:
            logger.info("Preloading semantic gate...")
            from src.agents.semantic_gate import get_semantic_gate

            gate = get_semantic_gate()
            set_semantic_gate(gate)

            logger.info("Semantic gate loaded successfully")
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
