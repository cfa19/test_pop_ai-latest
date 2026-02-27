"""
Runner Orchestrator — processes a single queue item through the full pipeline:
consent check → extraction → memory card creation via RPC.
"""

import logging

from src.config import Tables, get_supabase
from src.services import consent_manager, runner_extraction
from src.utils.harmonia_api import create_memory_proposal_rpc

logger = logging.getLogger(__name__)


async def process_queue_item(queue_item: dict) -> str:
    """
    Process a single queue item through the full pipeline.

    Args:
        queue_item: Row from memory_extraction_queue

    Returns:
        Final status: 'completed', 'failed', or 'skipped'
    """
    completion_id = queue_item["completion_id"]
    user_id = queue_item["user_id"]

    # 1. Fetch full completion record
    try:
        supabase = get_supabase()
        result = (
            supabase.table(Tables.ACTIVITY_COMPLETIONS)
            .select("*")
            .eq("id", completion_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            logger.warning(f"Completion {completion_id} not found")
            return "failed"
        completion = result.data[0]
    except Exception as e:
        logger.error(f"Failed to fetch completion {completion_id}: {e}")
        return "failed"

    # 2. Check consent (Layer 1 — fail-closed)
    if not consent_manager.check_consent(user_id):
        logger.info(f"No ai_training consent for user {user_id}, skipping")
        return "skipped"

    # 3. Extract proposals (Tiers 1-3)
    try:
        proposals = await runner_extraction.process_completion(completion)
        logger.info(f"Extracted {len(proposals)} proposals from completion {completion_id}")
    except Exception as e:
        logger.error(f"Extraction failed for completion {completion_id}: {e}")
        return "failed"

    if not proposals:
        logger.info(f"No proposals extracted from completion {completion_id}")
        return "completed"

    # 4. Write each proposal via RPC (Layer 3 consent check at DB level)
    written = 0
    for proposal in proposals:
        try:
            result = create_memory_proposal_rpc(user_id, proposal)
            if result:
                written += 1
        except Exception as e:
            logger.error(f"Failed to write proposal: {e}")

    logger.info(f"Written {written}/{len(proposals)} memory cards for completion {completion_id}")
    return "completed" if written > 0 else "failed"
