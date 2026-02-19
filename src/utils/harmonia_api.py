"""
Harmonia API Client - Store B (Memory Cards)

Stores LLM-extracted information as memory cards in Supabase (Store B).
Memory cards go through user validation before being applied to Store A.

Pipeline: AI extracts → memory_card (proposed) → user validates → applied to profile

Supports 5 context types:
- professional: position, experience, awards, aspirations, challenges, job search
- psychological: personality, values, motivations, confidence, stress
- learning: skills, education, gaps, aspirations, certifications, history
- social: mentors, mentees, network, networking
- personal: personal life, health, finances, goals, lifestyle, constraints
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from src.config import NEXT_PUBLIC_BASE_URL, Tables

logger = logging.getLogger(__name__)


# =============================================================================
# Entity → Memory Card Type mapping
# =============================================================================
# Maps entity/sub_entity names to memory card types (DB CHECK constraint).
# Valid types: competence, experience, preference, aspiration, trait, emotion, connection

_ENTITY_TO_CARD_TYPE: Dict[str, str] = {
    # professional entities
    "current_position": "competence",
    "professional_experience": "experience",
    "awards": "competence",
    "professional_aspirations": "aspiration",
    "workplace_challenges": "experience",
    "job_search_status": "experience",
    # psychological entities
    "personality_profile": "trait",
    "values": "preference",
    "motivations": "preference",
    "confidence_and_self_perception": "emotion",
    "stress_and_coping": "emotion",
    # learning entities
    "current_skills": "competence",
    "education_history": "experience",
    "learning_gaps": "aspiration",
    "learning_aspirations": "aspiration",
    "certifications": "competence",
    "learning_history": "experience",
    # social entities
    "mentors": "connection",
    "mentees": "connection",
    "professional_network": "connection",
    "networking": "connection",
    # personal entities
    "personal_life": "trait",
    "health_and_wellbeing": "emotion",
    "financial_situation": "trait",
    "personal_goals": "aspiration",
    "lifestyle_preferences": "preference",
    "life_constraints": "trait",
    # sub-entity level fallbacks
    "dream_roles": "aspiration",
    "compensation_expectations": "aspiration",
    "skills": "competence",
    "portfolio_items": "competence",
    "values_alignment": "preference",
    "stress": "emotion",
    "confidence": "emotion",
    "life_goals": "aspiration",
}


def _resolve_card_type(entity: Optional[str], sub_entity: str) -> str:
    """Resolve the memory card type from entity or sub_entity name."""
    if entity and entity in _ENTITY_TO_CARD_TYPE:
        return _ENTITY_TO_CARD_TYPE[entity]
    if sub_entity in _ENTITY_TO_CARD_TYPE:
        return _ENTITY_TO_CARD_TYPE[sub_entity]
    return "competence"


# =============================================================================
# Sub-entity → primary rawData field name
# =============================================================================
# When the LLM returns a simple value (not a dict), we wrap it with
# the correct field name instead of a generic "value".

_SUB_ENTITY_PRIMARY_FIELD: Dict[str, str] = {
    # professional
    "current_position": "title", "role": "role", "company": "company",
    "dream_roles": "desired_roles", "compensation_expectations": "target_salary",
    "desired_work_environment": "work_mode", "career_change_considerations": "considering_change",
    "job_search_status": "currently_searching", "awards": "awards",
    "workplace_challenges": "issueType",
    # learning
    "skills": "skill_name", "education_history": "degrees",
    "skill_gaps": "missing_skills", "knowledge_gaps": "missing_knowledge",
    "skill_aspirations": "target_skills", "education_aspirations": "desired_degrees",
    "certification_aspirations": "target_certs", "certifications": "earned_certs",
    "learning_history": "past_courses",
    # social
    "mentors": "mentor_name", "mentees": "mentee_name",
    "professional_network": "connections",
    "networking_activities": "activity_type", "networking_goals": "target_connections",
    "networking_preferences": "preferred_formats",
    # psychological
    "personality_profile": "personality_type", "values": "professional_values",
    "motivations": "intrinsic_motivations",
    "confidence_levels": "overall_confidence",
    "confidence_and_self_perception": "overall_confidence",
    "imposter_syndrome_and_doubt": "imposter_level",
    "self_talk_and_validation": "inner_critic_strength",
    "confidence_building_strategies": "strategies_that_help",
    "stress_and_coping": "stress_level",
    # personal
    "personal_life": "life_stage", "physical_health": "overall_health",
    "mental_health": "conditions", "addictions_or_recovery": "addiction_type",
    "overall_wellbeing": "wellbeing_score",
    "financial_situation": "stability", "personal_goals": "non_career_goals",
    "lifestyle_preferences": "work_life_balance",
    "life_constraints": "constraint_type",
    # new schemas
    "portfolio_items": "title", "values_alignment": "alignment_score",
    # common fallbacks
    "stress": "stress_level", "confidence": "confidence_level",
    "life_goals": "non_career_goals",
}


# =============================================================================
# Helpers
# =============================================================================

def _build_content(sub_entity: str, raw_data: Dict[str, Any]) -> str:
    """Build content as nested JSON for frontend collapsible tree rendering."""
    if not raw_data:
        return json.dumps({sub_entity: None}, ensure_ascii=False)

    clean = {k: v for k, v in raw_data.items() if v is not None and v != "" and v != []}
    if not clean:
        return json.dumps({sub_entity: None}, ensure_ascii=False)

    return json.dumps({sub_entity: clean}, ensure_ascii=False)


def _build_raw_data(sub_entity: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build rawData JSONB using taxonomy field names."""
    data = {k: v for k, v in extracted_data.items() if k not in ("content", "type")}

    if list(data.keys()) == ["value"]:
        field_name = _SUB_ENTITY_PRIMARY_FIELD.get(sub_entity, sub_entity)
        return {field_name: data["value"]}

    return data


# =============================================================================
# API Client (HTTP requests to Next.js)
# =============================================================================

def _make_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    user_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Make HTTP request to Harmonia Next.js API."""
    if not NEXT_PUBLIC_BASE_URL:
        logger.error("Harmonia API configuration missing (SUPABASE_URL or SUPABASE_KEY)")
        return None

    url = f"{NEXT_PUBLIC_BASE_URL}{endpoint}"
    request_headers = {
        "Authorization": user_token,
        "Content-Type": "application/json",
    }
    if headers:
        request_headers.update(headers)

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=request_headers, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, headers=request_headers, json=data, timeout=10)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=request_headers, json=data, timeout=10)
        elif method.upper() == "PATCH":
            response = requests.patch(url, headers=request_headers, json=data, timeout=10)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=request_headers, timeout=10)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        logger.error(f"Harmonia API timeout: {method} {endpoint}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Harmonia API error: {method} {endpoint} - {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                logger.error(f"Response body: {e.response.text}")
            except Exception:
                pass
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling Harmonia API: {e}")
        return None


# =============================================================================
# Profile Management
# =============================================================================

def check_profile_exists(user_token: str) -> bool:
    """Check if user's Canonical Profile exists."""
    result = _make_request("GET", "/api/harmonia/canonical/profile", user_token=user_token)
    if result and "data" in result:
        logger.info("User profile exists")
        return True
    logger.info("User profile does not exist")
    return False


def create_profile(user_token: str, stage: str = "EXPLORATION") -> bool:
    """Create a new Canonical Profile for the user."""
    result = _make_request(
        "POST", "/api/harmonia/canonical/profile",
        data={"stage": stage}, user_token=user_token,
    )
    if result and "data" in result:
        logger.info(f"Created user profile with stage: {stage}")
        return True
    logger.error("Failed to create user profile")
    return False


def ensure_profile_exists(user_token: str) -> bool:
    """Ensure user's Canonical Profile exists, creating it if needed."""
    if check_profile_exists(user_token):
        return True
    logger.info("Profile does not exist, creating...")
    return create_profile(user_token)


# =============================================================================
# Main Storage Function — Memory Cards (Store B)
# =============================================================================

def store_extracted_information(
    supabase,
    category: str,
    subcategory: str,
    extracted_data: Dict[str, Any],
    user_id: str,
    entity: Optional[str] = None,
    conversation_id: Optional[str] = None,
    confidence: float = 0.85,
) -> Dict[str, Any]:
    """
    Store extracted information as a memory card in Supabase (Store B).

    Inserts into the `memory_cards` table following POP-507 v2 schema.
    Cards are created with status 'proposed' — user validates before
    they are applied to the canonical profile (Store A).

    Args:
        supabase: Supabase client (service role)
        category: Context name (e.g., "professional", "learning")
        subcategory: Sub-entity name (e.g., "dream_roles", "skills")
        extracted_data: Dict with extracted fields from LLM
        user_id: User UUID
        entity: Entity name (e.g., "professional_aspirations")
        conversation_id: Conversation UUID for source tracing
        confidence: Classifier confidence score (0-1)

    Returns:
        Dict with "success", "context", "resource", and "created_ids"
    """
    logger.info(f"Storing memory card: {category}/{entity}/{subcategory}")

    now = datetime.now(timezone.utc).isoformat()
    card_id = str(uuid.uuid4())

    card_type = _resolve_card_type(entity, subcategory)
    raw_data = _build_raw_data(subcategory, extracted_data)
    content = _build_content(subcategory, raw_data)

    source = {
        "type": "coach",
        "sourceId": conversation_id or "unknown",
        "extractedAt": now,
    }

    try:
        (
            supabase.table(Tables.MEMORY_CARDS)
            .insert({
                "id": card_id,
                "user_id": user_id,
                "content": content,
                "type": card_type,
                "confidence": round(min(confidence, 1.0), 2),
                "source": source,
                "status": "proposed",
                "tags": [category, subcategory],
                "raw_data": raw_data,
            })
            .execute()
        )

        logger.info(f"Created memory card {card_id}: type={card_type} content='{content[:60]}'")
        return {
            "success": True,
            "context": category,
            "resource": subcategory,
            "created_ids": [card_id],
        }

    except Exception as e:
        logger.error(f"Failed to insert memory card: {e}")
        return {"success": False, "error": str(e)}
