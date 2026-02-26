"""
Harmonia Memory Card Storage

Stores extracted information as memory cards directly in Supabase.
Each extraction (category + subcategory) becomes a memory card with
status "proposed" for user validation in the Harmonia UI.

Memory card types (from EXTRACTION_SCHEMAS):
  competence, experience, preference, aspiration, trait, emotion, connection
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.config import get_supabase

logger = logging.getLogger(__name__)

# # =============================================================================
# # Alternative: Harmonia Next.js API (uncomment if using trajectoire frontend)
# # =============================================================================
# # Set NEXT_PUBLIC_BASE_URL to the trajectoire deployment URL to use this path.
# # Memory cards will be created via POST /api/harmonia/journal/memory-cards
# # which goes through the Next.js API layer with full validation.
# #
# # from src.config import NEXT_PUBLIC_BASE_URL
# # import requests
# #
# # def _store_via_harmonia_api(category, subcategory, extracted_data, user_id, user_token):
# #     url = f"{NEXT_PUBLIC_BASE_URL}/api/harmonia/journal/memory-cards"
# #     response = requests.post(url, json={
# #         "content": extracted_data.get("content"),
# #         "type": extracted_data.get("type"),
# #         "confidence": 0.9,
# #         "source": {"type": "coach", "sourceId": "pop-ai",
# #                     "extractedAt": datetime.now(timezone.utc).isoformat()},
# #         "tags": [category, subcategory],
# #     }, headers={"Authorization": user_token}, timeout=10)
# #     return response.json()
# # =============================================================================


def store_extracted_information(
    category: str,
    subcategory: str,
    extracted_data: Dict[str, Any],
    user_id: str,
    user_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Store extracted information as a memory card directly in Supabase.

    Inserts into the `memory_cards` table using the service_role client
    (bypasses RLS). The card starts with status "proposed" so the user
    can validate or reject it in the Harmonia UI.

    Args:
        category: Primary category (e.g. "professional", "learning")
        subcategory: Subcategory / entity name (e.g. "work_history")
        extracted_data: Dict with "content" (JSON str) and "type" keys
        user_id: User UUID
        user_token: User JWT token (unused for direct Supabase, kept for compat)

    Returns:
        Dict with "success", "context", "resource", and "created_ids"
    """
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
        raw_data = fields  # raw_data keeps the flat extracted fields
        content = json.dumps({subcategory: fields})
    except (json.JSONDecodeError, TypeError):
        content = raw_content

    try:
        supabase = get_supabase()
    except Exception as e:
        logger.error(f"Supabase client not available: {e}")
        return {"success": False, "error": "Supabase not configured"}

    now = datetime.now(timezone.utc).isoformat()

    row = {
        "user_id": user_id,
        "content": content,
        "type": card_type,
        "confidence": 0.9,
        "source": {
            "type": "coach",
            "sourceId": "pop-ai",
            "extractedAt": now,
        },
        "status": "proposed",
        "tags": [category, subcategory],
        "linked_contexts": [],
        "raw_data": raw_data,
        "applied_field_paths": [],
        "mapping_attempts": 0,
    }

    try:
        result = supabase.table("memory_cards").insert(row).execute()

        if result.data:
            card_id = result.data[0].get("id")
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
        else:
            logger.error(
                f"Supabase insert returned no data for "
                f"{category}.{subcategory}"
            )
            return {"success": False, "error": "Insert returned no data"}

    except Exception as e:
        logger.error(
            f"Failed to create memory card for "
            f"{category}.{subcategory}: {e}"
        )
        return {"success": False, "error": str(e)}
