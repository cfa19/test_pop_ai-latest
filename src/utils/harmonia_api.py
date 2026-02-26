"""
Harmonia Memory Card Storage

Stores extracted information as memory cards via the Harmonia (trajectoire)
Next.js API. Each extraction (category + subcategory) becomes a memory card
with status "proposed" for user validation in the Harmonia UI.

The API endpoint POST /api/harmonia/journal/memory-cards validates the
payload with Zod and inserts into Supabase with proper authentication.

Memory card types (from EXTRACTION_SCHEMAS):
  competence, experience, preference, aspiration, trait, emotion, connection
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from src.config import NEXT_PUBLIC_BASE_URL

logger = logging.getLogger(__name__)

# =============================================================================
# Direct Supabase fallback (uncomment if trajectoire is not deployed)
# =============================================================================
# from src.config import get_supabase
#
# def _store_via_supabase(category, subcategory, content, card_type, raw_data, user_id):
#     """Insert memory card directly into Supabase (bypasses RLS)."""
#     supabase = get_supabase()
#     now = datetime.now(timezone.utc).isoformat()
#     row = {
#         "user_id": user_id,
#         "content": content,
#         "type": card_type,
#         "confidence": 0.9,
#         "source": {"type": "coach", "sourceId": "pop-ai", "extractedAt": now},
#         "status": "proposed",
#         "tags": [category, subcategory],
#         "linked_contexts": [],
#         "raw_data": raw_data,
#         "applied_field_paths": [],
#         "mapping_attempts": 0,
#     }
#     result = supabase.table("memory_cards").insert(row).execute()
#     if result.data:
#         return {"success": True, "created_ids": [result.data[0].get("id")]}
#     return {"success": False, "error": "Insert returned no data"}
# =============================================================================


def store_extracted_information(
    category: str,
    subcategory: str,
    extracted_data: Dict[str, Any],
    user_id: str,
    user_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Store extracted information as a memory card via the Harmonia Next.js API.

    POST /api/harmonia/journal/memory-cards with Zod-validated payload.
    Requires NEXT_PUBLIC_BASE_URL and a valid user JWT token.

    Args:
        category: Primary category (e.g. "professional", "learning")
        subcategory: Subcategory / entity name (e.g. "work_history")
        extracted_data: Dict with "content" (JSON str) and "type" keys
        user_id: User UUID
        user_token: User JWT token (Authorization header value)

    Returns:
        Dict with "success", "context", "resource", and "created_ids"
    """
    if not NEXT_PUBLIC_BASE_URL:
        logger.debug(
            "Store Information: NEXT_PUBLIC_BASE_URL not configured, skipping"
        )
        return {"success": False, "error": "NEXT_PUBLIC_BASE_URL not configured"}

    if not user_token:
        logger.warning("Store Information: No user token provided, skipping")
        return {"success": False, "error": "No user token"}

    raw_content = extracted_data.get("content")
    card_type = extracted_data.get("type")

    if not raw_content or not card_type:
        logger.warning(
            f"Skipping memory card for {category}.{subcategory}: "
            "missing content or type"
        )
        return {"success": False, "error": "Missing content or type"}

    # Parse raw content and build display content wrapped with subcategory
    # so the tree view in the UI shows e.g. "work_history { role: ..., company: ... }"
    raw_data = None
    try:
        fields = json.loads(raw_content)
        # Remove meta fields that aren't user data
        fields.pop("content", None)
        fields.pop("type", None)
        raw_data = fields  # flat extracted fields for rawData
        content = json.dumps({subcategory: fields})
    except (json.JSONDecodeError, TypeError):
        content = raw_content

    now = datetime.now(timezone.utc).isoformat()

    # Payload matching trajectoire's createMemoryProposalSchema (Zod)
    payload = {
        "content": content,
        "type": card_type,
        "confidence": 0.9,
        "source": {
            "type": "coach",
            "sourceId": "pop-ai",
            "extractedAt": now,
        },
        "rawData": raw_data,
        "tags": [category, subcategory],
    }

    url = f"{NEXT_PUBLIC_BASE_URL}/api/harmonia/journal/memory-cards"

    # Ensure token has Bearer prefix
    auth_header = user_token
    if not auth_header.startswith("Bearer "):
        auth_header = f"Bearer {auth_header}"

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": auth_header},
            timeout=10,
        )

        if response.status_code == 201:
            data = response.json()
            card = data.get("data", {})
            card_id = card.get("id")
            logger.info(
                f"Created memory card {card_id} "
                f"for {category}.{subcategory} ({card_type})"
            )
            return {
                "success": True,
                "context": category,
                "resource": subcategory,
                "created_ids": [card_id],
            }

        # Handle error responses
        error_body = {}
        try:
            error_body = response.json()
        except Exception:
            pass

        error_msg = error_body.get("error", f"HTTP {response.status_code}")
        details = error_body.get("details", "")

        logger.error(
            f"Failed to create memory card for {category}.{subcategory}: "
            f"{error_msg} {details}"
        )
        return {"success": False, "error": error_msg}

    except requests.exceptions.Timeout:
        logger.error(
            f"Timeout creating memory card for {category}.{subcategory}"
        )
        return {"success": False, "error": "Request timeout"}

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Connection error creating memory card for "
            f"{category}.{subcategory} (is trajectoire running?)"
        )
        return {"success": False, "error": "Connection error"}

    except Exception as e:
        logger.error(
            f"Failed to create memory card for "
            f"{category}.{subcategory}: {e}"
        )
        return {"success": False, "error": str(e)}
