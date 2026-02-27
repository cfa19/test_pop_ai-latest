"""
Consent Manager — checks ai_training consent before memory card extraction.

Read-only: only queries user_consents table, never writes to it.
Fail-closed: returns False on any error or missing consent.
"""

import logging
import time

from src.config import Tables, get_supabase

logger = logging.getLogger(__name__)

# TTL cache: {user_id: (consent_given, timestamp)}
_consent_cache: dict[str, tuple[bool, float]] = {}
_CACHE_TTL = 60.0  # seconds


def check_consent(user_id: str) -> bool:
    """
    Check if user has active ai_training consent.
    Uses TTL cache (60s) to reduce DB load during burst completions.
    Fail-closed: returns False on any error.
    """
    # Check cache first
    cached = _consent_cache.get(user_id)
    if cached and (time.time() - cached[1]) < _CACHE_TTL:
        return cached[0]

    try:
        supabase = get_supabase()
        result = (
            supabase.table(Tables.USER_CONSENTS)
            .select("consent_given")
            .eq("user_id", user_id)
            .eq("consent_type", "ai_training")
            .eq("consent_given", True)
            .is_("revoked_at", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        has_consent = bool(result.data)
        _consent_cache[user_id] = (has_consent, time.time())

        if not has_consent:
            logger.info(f"No ai_training consent for user {user_id}")

        return has_consent

    except Exception as e:
        logger.error(f"Consent check failed for user {user_id}: {e}")
        # Fail-closed: no consent on error
        return False


def invalidate_consent_cache(user_id: str) -> None:
    """Clear cached consent for a user (e.g. after revocation)."""
    _consent_cache.pop(user_id, None)
