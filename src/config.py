"""
Centralized configuration for the Pop Skills AI platform.

All environment variables, constants, and client factories live here.
Other modules import from this file instead of reading env vars directly.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client
from voyageai.client import Client as VoyageAI

# Load .env once at import time
load_dotenv()

# =============================================================================
# Environment Variables
# =============================================================================

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) must be set in environment variables")

SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_SERVICE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set in environment variables")

# NextJS Frontend
NEXT_PUBLIC_BASE_URL = os.getenv("NEXT_PUBLIC_BASE_URL")

# OpenAI (chat + LLM classifier)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

# Groq (fast chat, OpenAI-compatible API)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Voyage AI (all embeddings)
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY must be set in environment variables")

# Provider + Model configuration
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "openai")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "voyage")
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3-large")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ONNX token classifier path
ONNX_MODELS_PATH = os.getenv("ONNX_MODELS_PATH", "training/models_onnx")

# --- Semantic gate (disabled — token classifier handles off-topic via O label) ---
# SEMANTIC_GATE_ENABLED = os.getenv("SEMANTIC_GATE_ENABLED", "false").lower() in ("true", "1", "yes")
# SEMANTIC_GATE_MODEL = os.getenv("SEMANTIC_GATE_MODEL", "all-MiniLM-L6-v2")
# SEMANTIC_GATE_TUNING_PATH = os.getenv("SEMANTIC_GATE_TUNING_PATH", "training/results/semantic_gate_tuning.json")
# SEMANTIC_GATE_LOCAL_FILES_ONLY = os.getenv("SEMANTIC_GATE_LOCAL_FILES_ONLY", "false").lower() in ("true", "1", "yes")
# SEMANTIC_GATE_MODEL_PATH = os.getenv("SEMANTIC_GATE_MODEL_PATH", "training/models/sentence_transformers")

# --- Legacy PyTorch classifier paths (replaced by ONNX_MODELS_PATH) ---
# PRIMARY_INTENT_CLASSIFIER_TYPE = os.getenv("PRIMARY_INTENT_CLASSIFIER_TYPE", "openai")
# SECONDARY_INTENT_CLASSIFIER_TYPE = os.getenv("SECONDARY_INTENT_CLASSIFIER_TYPE", "openai")
# INTENT_CLASSIFIER_MODEL_PATH = os.getenv("INTENT_CLASSIFIER_MODEL_PATH", "training/models/latest")
# SPAN_CLASSIFIER_MODEL_PATH = os.getenv("SPAN_CLASSIFIER_MODEL_PATH", "")

# Language detection (optional FastText model for redundancy)
LANG_DETECT_FASTTEXT_MODEL_PATH = os.getenv("LANG_DETECT_FASTTEXT_MODEL_PATH", "")  # e.g. "data/lid.176.bin"

# Allowed languages (ISO 639-1 codes). Only these are considered; others fall back to "en".
# Comma-separated, e.g. "en,es,fr,de,pt,it,nl,pl,ru,ar,zh,ja,ko,hi". "en" is always included.
_LANG_DETECT_ALLOWED_RAW = os.getenv("LANG_DETECT_ALLOWED_LANGUAGES", "en,es,fr")
LANG_DETECT_ALLOWED_LANGUAGES: frozenset[str] = frozenset(
    c.strip().lower()[:2] for c in _LANG_DETECT_ALLOWED_RAW.split(",") if c.strip()
) | frozenset({"en"})
# Map language code to full name
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "et": "Estonian",
}

# Verbose mode (set by CLI flag -v)
VERBOSE_MODE = False

# =============================================================================
# Constants
# =============================================================================

RRF_K = 60


class Tables:
    """Database table names."""

    CONVERSATION_HISTORY = "conversation_history"
    RETRIEVAL_CHUNKS = "retrieval_chunks"
    GENERAL_EMBEDDINGS_1024 = "general_embeddings_1024"
    USER_EMBEDDINGS_1024 = "user_embeddings_1024"


class RPCFunctions:
    """Supabase RPC function names."""

    RAG_SEARCH_SEMANTIC = "rag_search_semantic"
    RAG_SEARCH_FULLTEXT = "rag_search_fulltext"
    RAG_HYBRID_SEARCH = "rag_hybrid_search_user_context"
    SEARCH_CONVERSATION_HISTORY = "search_conversation_history"
    SEARCH_USER_CONTEXT_CHUNKS = "search_user_context_chunks"


class SourceTypes:
    """Source type discriminators for retrieval_chunks."""

    CONVERSATION_MESSAGE = "conversation_message"


# =============================================================================
# Client Factories (lazy, cached)
# =============================================================================


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Create and cache the Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@lru_cache(maxsize=1)
def get_openai() -> OpenAI:
    """Create and cache the OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)


@lru_cache(maxsize=2)
def get_client_by_provider(provider: str) -> OpenAI:
    """
    Get embedding client based on provider name. Cached to avoid recreating clients.

    Args:
        provider: Provider name ('openai' or 'voyage')

    Returns:
        OpenAI or VoyageAI client for the specified provider
    """
    if provider == "openai":
        return OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
    elif provider == "groq":
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", timeout=30.0)
    elif provider == "voyage":
        return VoyageAI(api_key=VOYAGE_API_KEY, timeout=30.0)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Must be 'openai', 'groq', or 'voyage'.")


# =============================================================================
# Model Instances (preloaded on startup)
# =============================================================================

_hierarchical_classifier_instance = None


def set_hierarchical_classifier(classifier):
    """Set the preloaded hierarchical ONNX classifier (called during startup)."""
    global _hierarchical_classifier_instance
    _hierarchical_classifier_instance = classifier


def get_hierarchical_classifier():
    """
    Get the preloaded hierarchical ONNX classifier.

    Returns None if not preloaded (will fall back to lazy loading in workflow).
    """
    return _hierarchical_classifier_instance


# --- Semantic gate singleton (disabled) ---
# _semantic_gate_instance = None
#
# def set_semantic_gate(gate):
#     global _semantic_gate_instance
#     _semantic_gate_instance = gate
#
# def get_semantic_gate_instance():
#     return _semantic_gate_instance

# --- Legacy PyTorch span classifier singletons (replaced by ONNX) ---
# _intent_classifier_instance = None
# _primary_span_classifier_instance = None
# _secondary_span_classifier_instances: dict = {}
#
# def set_intent_classifier(classifier):
#     global _intent_classifier_instance
#     _intent_classifier_instance = classifier
#
# def get_intent_classifier():
#     return _intent_classifier_instance
#
# def set_primary_span_classifier(classifier):
#     global _primary_span_classifier_instance
#     _primary_span_classifier_instance = classifier
#
# def get_primary_span_classifier():
#     return _primary_span_classifier_instance
#
# def set_secondary_span_classifier(category, classifier):
#     global _secondary_span_classifier_instances
#     _secondary_span_classifier_instances[category] = classifier
#
# def get_secondary_span_classifier(category):
#     return _secondary_span_classifier_instances.get(category)
