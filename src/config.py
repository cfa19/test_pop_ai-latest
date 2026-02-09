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

# OpenAI (chat + LLM classifier)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

# Voyage AI (all embeddings)
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY must be set in environment variables")

# Model configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-3-large")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Intent classifier settings
PRIMARY_INTENT_CLASSIFIER_TYPE = os.getenv("PRIMARY_INTENT_CLASSIFIER_TYPE", "openai")  # "openai", "onnx", or "bert"
SECONDARY_INTENT_CLASSIFIER_TYPE = os.getenv("SECONDARY_INTENT_CLASSIFIER_TYPE", "openai")  # "openai" or "bert"
INTENT_CLASSIFIER_MODEL_PATH = os.getenv("INTENT_CLASSIFIER_MODEL_PATH", "training/models/latest")
ONNX_HIERARCHY_PATH = os.getenv("ONNX_HIERARCHY_PATH", "training/models/onnx")

# Semantic gate settings (Stage 1 filtering)
SEMANTIC_GATE_ENABLED = os.getenv("SEMANTIC_GATE_ENABLED", "true").lower() in ("true", "1", "yes")
SEMANTIC_GATE_MODEL = os.getenv("SEMANTIC_GATE_MODEL", "all-MiniLM-L6-v2")
SEMANTIC_GATE_TUNING_PATH = os.getenv("SEMANTIC_GATE_TUNING_PATH", "training/results/semantic_gate_hierarchical_tuning.json")
SEMANTIC_GATE_ONNX_MODEL_PATH = os.getenv("SEMANTIC_GATE_ONNX_MODEL_PATH", "training/models/onnx/semantic_gate")
SEMANTIC_GATE_CENTROIDS_DIR = os.getenv("SEMANTIC_GATE_CENTROIDS_DIR", "training/models/onnx/semantic_gate")

# HuggingFace Hub (model downloads for deployment)
HF_REPO_ID = os.getenv("HF_REPO_ID", "cfa0819/pop-skills-onnx")

# Groq (fast entity extraction + chat - optional)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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


@lru_cache(maxsize=1)
def get_groq() -> OpenAI | None:
    """Create and cache the Groq client (OpenAI-compatible). Returns None if no API key."""
    if not GROQ_API_KEY:
        return None
    return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", timeout=30.0)


@lru_cache(maxsize=3)
def get_client_by_provider(provider: str) -> OpenAI:
    """
    Get client based on provider name. Cached to avoid recreating clients.

    Args:
        provider: Provider name ('openai', 'voyage', or 'groq')

    Returns:
        OpenAI, VoyageAI, or Groq client for the specified provider
    """
    if provider == "openai":
        return OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
    elif provider == "voyage":
        return VoyageAI(api_key=VOYAGE_API_KEY, timeout=30.0)
    elif provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY must be set to use Groq as chat provider")
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", timeout=30.0)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Must be 'openai', 'voyage', or 'groq'.")


# =============================================================================
# Model Instances (preloaded on startup)
# =============================================================================

_intent_classifier_instance = None
_semantic_gate_instance = None


def set_intent_classifier(classifier):
    """Set the preloaded intent classifier instance (called during startup)."""
    global _intent_classifier_instance
    _intent_classifier_instance = classifier


def get_intent_classifier():
    """
    Get the preloaded intent classifier instance.

    Returns None if not preloaded (will fall back to lazy loading in workflow).
    """
    return _intent_classifier_instance


def set_semantic_gate(gate):
    """Set the preloaded semantic gate instance (called during startup)."""
    global _semantic_gate_instance
    _semantic_gate_instance = gate


def get_semantic_gate_instance():
    """
    Get the preloaded semantic gate instance.

    Returns None if not preloaded (will fall back to lazy loading in workflow).
    """
    return _semantic_gate_instance
