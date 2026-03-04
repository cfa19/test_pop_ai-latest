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

# Load .envlocal for local development
load_dotenv(".envlocal")

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

# Anthropic (optional — required when CHAT_MODEL starts with "claude")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Google AI (optional — required when CHAT_MODEL starts with "gemini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Voyage AI (all embeddings)
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY must be set in environment variables")

# Model configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-4-large")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Extraction provider (OpenAI gpt-4o-mini for structured JSON — used by runner_extraction)
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")

# Temperature settings
TEMPERATURE_TRANSLATION = float(os.getenv("TEMPERATURE_TRANSLATION", "0.3"))
TEMPERATURE_EXTRACTION = float(os.getenv("TEMPERATURE_EXTRACTION", "0.1"))
TEMPERATURE_CLASSIFICATION = float(os.getenv("TEMPERATURE_CLASSIFICATION", "0.0"))
TEMPERATURE_RESPONSE = float(os.getenv("TEMPERATURE_RESPONSE", "0.3"))
TEMPERATURE_TASK_RESPONSE = float(os.getenv("TEMPERATURE_TASK_RESPONSE", "0.5"))
TEMPERATURE_RECOMMENDATION = float(os.getenv("TEMPERATURE_RECOMMENDATION", "0.7"))

# RAG search parameters
RAG_DOC_TOP_K = int(os.getenv("RAG_DOC_TOP_K", "3"))
RAG_CONVERSATION_TOP_K = int(os.getenv("RAG_CONVERSATION_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.6"))

# Profile recap search parameters (broader retrieval for "what do you know about me")
PROFILE_RECAP_TOP_K = int(os.getenv("PROFILE_RECAP_TOP_K", "15"))
PROFILE_RECAP_SIMILARITY_THRESHOLD = float(os.getenv("PROFILE_RECAP_SIMILARITY_THRESHOLD", "0.3"))

# Max tokens for LLM responses
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# Background worker: number of historical messages to process per batch
HISTORICAL_BATCH_SIZE = int(os.getenv("HISTORICAL_BATCH_SIZE", "50"))

# Client timeout (seconds)
CLIENT_TIMEOUT = float(os.getenv("CLIENT_TIMEOUT", "30.0"))

# ONNX token classifier path
ONNX_MODELS_PATH = os.getenv("ONNX_MODELS_PATH", "training/models_onnx")

# Semantic gate settings (Stage 1 filtering)
SEMANTIC_GATE_ENABLED = os.getenv("SEMANTIC_GATE_ENABLED", "true").lower() in ("true", "1", "yes")
SEMANTIC_GATE_MODEL = os.getenv("SEMANTIC_GATE_MODEL", "all-MiniLM-L6-v2")
SEMANTIC_GATE_TUNING_PATH = os.getenv("SEMANTIC_GATE_TUNING_PATH", "training/results/semantic_gate_tuning.json")
# Use only cached model (no network). Set to "true" for offline; model must be in cache_folder first.
SEMANTIC_GATE_LOCAL_FILES_ONLY = os.getenv("SEMANTIC_GATE_LOCAL_FILES_ONLY", "false").lower() in ("true", "1", "yes")
SEMANTIC_GATE_MODEL_PATH = os.getenv("SEMANTIC_GATE_MODEL_PATH", "training/models/sentence_transformers")

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

# Runner extraction (memory cards from activity completions) — disabled for now
RUNNER_EXTRACTION_ENABLED = os.getenv("RUNNER_EXTRACTION_ENABLED", "false").lower() in ("true", "1", "yes")
QUEUE_POLL_INTERVAL = float(os.getenv("QUEUE_POLL_INTERVAL", "5.0"))
QUEUE_BATCH_SIZE = int(os.getenv("QUEUE_BATCH_SIZE", "10"))
QUEUE_MAX_RETRIES = int(os.getenv("QUEUE_MAX_RETRIES", "3"))

# Verbose mode (set by CLI flag -v)
VERBOSE_MODE = False

# =============================================================================
# Constants
# =============================================================================

RRF_K = 60

VALID_CARD_TYPES = frozenset({
    "competence", "experience", "preference", "aspiration",
    "trait", "emotion", "connection",
})


class Tables:
    """Database table names."""

    CONVERSATION_HISTORY = "conversation_history"
    RETRIEVAL_CHUNKS = "retrieval_chunks"
    GENERAL_EMBEDDINGS_1024 = "general_embeddings_1024"
    USER_EMBEDDINGS_1024 = "user_embeddings_1024"
    # Runner extraction tables (not used yet)
    USER_CONSENTS = "user_consents"
    MEMORY_EXTRACTION_QUEUE = "memory_extraction_queue"
    MEMORY_CARDS = "memory_cards"
    ACTIVITY_COMPLETIONS = "activity_completions"


class RPCFunctions:
    """Supabase RPC function names."""

    RAG_SEARCH_SEMANTIC = "rag_search_semantic"
    RAG_SEARCH_FULLTEXT = "rag_search_fulltext"
    RAG_HYBRID_SEARCH = "rag_hybrid_search_user_context"
    SEARCH_CONVERSATION_HISTORY = "search_conversation_history"
    SEARCH_USER_CONTEXT_CHUNKS = "search_user_context_chunks"
    SEARCH_RUNNER_CHUNKS = "search_runner_chunks"
    # Runner extraction RPCs (not used yet)
    CLAIM_EXTRACTION_BATCH = "claim_extraction_batch"
    CREATE_MEMORY_PROPOSAL = "create_memory_proposal"


class SourceTypes:
    """Source type discriminators for retrieval_chunks."""

    CONVERSATION_MESSAGE = "conversation_message"


# =============================================================================
# Utility Functions
# =============================================================================


def detect_provider(model_name: str) -> str:
    """Infer the API provider from a model name.

    * ``claude-*``  -> anthropic
    * ``gemini-*``  -> google
    * ``voyage-*``  -> voyage
    * everything else -> openai
    """
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("gemini"):
        return "google"
    if model_name.startswith("voyage"):
        return "voyage"
    return "openai"


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
    return OpenAI(api_key=OPENAI_API_KEY, timeout=CLIENT_TIMEOUT)


@lru_cache(maxsize=4)
def get_client_by_provider(provider: str):
    """
    Get a cached client for the given provider.

    Supported providers:
      openai    — OpenAI chat + embeddings  (openai SDK)
      voyage    — Voyage AI embeddings      (voyageai SDK)
      anthropic — Anthropic chat            (anthropic SDK, optional)
      google    — Google Gemini chat        (google-generativeai SDK, optional)

    The anthropic and google SDKs are optional; an ImportError is raised with
    an install hint if they are not available.
    """
    if provider == "openai":
        return OpenAI(api_key=OPENAI_API_KEY, timeout=CLIENT_TIMEOUT)

    if provider == "voyage":
        return VoyageAI(api_key=VOYAGE_API_KEY, timeout=CLIENT_TIMEOUT)

    if provider == "anthropic":
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in the environment.")
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if provider == "google":
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "google-generativeai SDK not installed. Run: pip install google-generativeai"
            ) from exc
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in the environment.")
        genai.configure(api_key=GOOGLE_API_KEY)
        return genai

    raise ValueError(
        f"Unsupported provider: '{provider}'. "
        "Must be one of: 'openai', 'voyage', 'anthropic', 'google'."
    )
