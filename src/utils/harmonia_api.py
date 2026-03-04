"""
Harmonia API Client - Store A (Canonical Profile)

Utility functions for interacting with the Harmonia V3.1 API endpoints
for storing extracted information in the Canonical Profile (Store A) contexts.

Supports 6 context types:
- Professional: skills, experiences, certifications, current position
- Psychological: personality, values, motivations, working styles
- Learning: knowledge graph, skill gaps, courses, goals, preferred formats
- Social: mentors, peers, testimonials
- Emotional: confidence, energy patterns, stress triggers, celebrations
- Aspirational: dream roles, salary expectations, life goals, value alignment
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import httpx

from src.config import CLIENT_TIMEOUT, NEXT_PUBLIC_BASE_URL

logger = logging.getLogger(__name__)


class HarmoniaAPIError(Exception):
    """Raised when a Harmonia API request fails."""


# =============================================================================
# Configuration
# =============================================================================

# Base path for Canonical Profile API
CANONICAL_BASE = "/api/harmonia/canonical/contexts"

# Default confidence for AI-extracted memory cards (0.0–1.0)
EXTRACTION_CONFIDENCE = 0.9

# Default rating for extracted items on a 1–5 scale (used when LLM doesn't provide one)
DEFAULT_RATING = 3

# =============================================================================
# Subcategory to Endpoint Mapping
# =============================================================================

# Maps extraction subcategories to (context, resource, transform_function)
SUBCATEGORY_ENDPOINTS = {
    # Aspirational context
    "dream_roles": ("aspirational", "dream-roles", "transform_dream_roles"),
    "salary_expectations": ("aspirational", "salary-expectations", "transform_salary_expectations"),
    "life_goals": ("aspirational", "life-goals", "transform_life_goals"),
    "impact_legacy": ("aspirational", "life-goals", "transform_impact_legacy"),
    # Professional context
    "skills": ("professional", "skills", "transform_skills"),
    "experiences": ("professional", "experiences", "transform_experiences"),
    "certifications": ("professional", "certifications", "transform_certifications"),
    "current_position": ("professional", "experiences", "transform_current_position"),
    # Psychological context
    "personality_profile": ("psychological", "personality", "transform_personality"),
    "strengths": ("psychological", "personality", "transform_strengths"),
    "weaknesses": ("psychological", "personality", "transform_weaknesses"),
    "motivations": ("psychological", "motivations", "transform_motivations"),
    "work_style": ("psychological", "working-styles", "transform_work_style"),
    "values": ("psychological", "values", "transform_values"),
    # Learning context
    "knowledge": ("learning", "knowledge-graph", "transform_knowledge"),
    "learning_velocity": ("learning", None, "transform_learning_velocity"),  # No specific endpoint, update context
    "preferred_format": ("learning", "preferred-formats", "transform_preferred_format"),
    # Social context
    "mentors": ("social", "mentors", "transform_mentors"),
    "journey_peers": ("social", "peers", "transform_peers"),
    "people_helped": ("social", "peers", "transform_people_helped"),  # Store as peers with type="mentee"
    "testimonials": ("social", "testimonials", "transform_testimonials"),
    # Emotional context
    "confidence": ("emotional", "confidence", "transform_confidence"),
    "stress": ("emotional", "stress-triggers", "transform_stress"),
    "wellbeing": ("emotional", None, "transform_wellbeing"),  # No specific endpoint, update context
    "resilience": ("emotional", "celebrations", "transform_resilience"),
}

# =============================================================================
# API Client Functions
# =============================================================================


async def _make_request(
    method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make async HTTP request to Harmonia API.

    Args:
        method: HTTP method (GET, POST, PATCH, DELETE, PUT)
        endpoint: API endpoint path
        data: Request body data
        headers: Additional headers
        user_id: User UUID

    Returns:
        Response JSON

    Raises:
        HarmoniaAPIError: On any request failure
    """
    if not NEXT_PUBLIC_BASE_URL:
        raise HarmoniaAPIError("Harmonia API configuration missing (NEXT_PUBLIC_BASE_URL)")

    url = f"{NEXT_PUBLIC_BASE_URL}{endpoint}"

    request_headers = {
        "x-user-id": user_id,
        "Content-Type": "application/json",
    }

    if headers:
        request_headers.update(headers)

    try:
        async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
            response = await client.request(
                method.upper(),
                url,
                headers=request_headers,
                json=data,
            )
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException as e:
        raise HarmoniaAPIError(f"Timeout: {method} {endpoint}") from e
    except httpx.HTTPStatusError as e:
        body = e.response.text[:200] if e.response else ""
        raise HarmoniaAPIError(f"{method} {endpoint} failed: {e} | body: {body}") from e
    except httpx.HTTPError as e:
        raise HarmoniaAPIError(f"{method} {endpoint} failed: {e}") from e


# =============================================================================
# Data Transform Helpers
# =============================================================================


def _get_list_field(data: Dict[str, Any], *keys) -> list:
    """Get a list field from data dict, trying multiple keys. Splits comma-separated strings."""
    for key in keys:
        value = data.get(key)
        if value:
            if isinstance(value, str):
                return [item.strip() for item in value.split(",")]
            return value
    return []


# =============================================================================
# Data Transform Functions
# =============================================================================


def transform_dream_roles(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted dream_roles to API format."""
    items = []

    titles = _get_list_field(data, "occupations", "job_titles")

    for title in titles:
        if title:
            items.append({"title": str(title), "progress": 0})

    return items


def transform_salary_expectations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted salary_expectations to API format."""
    # Salary expectations might be a single object
    amount = data.get("salary_amount")
    range_val = data.get("salary_range")

    if amount or range_val:
        return [{"amount": str(amount or range_val), "timeframe": data.get("timeframe", "annual")}]
    return []


def transform_life_goals(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted life_goals to API format."""
    items = []
    goals = _get_list_field(data, "life_goals", "lifestyle_aspirations")

    for goal in goals:
        if goal:
            items.append({"title": str(goal), "progress": 0})

    return items


def transform_impact_legacy(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted impact_legacy to API format (as life goals)."""
    items = []
    impacts = _get_list_field(data, "impact_goals", "legacy_aspirations")

    for impact in impacts:
        if impact:
            items.append({"title": f"Impact: {impact}", "progress": 0})

    return items


def transform_skills(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted skills to API format."""
    items = []
    skills = _get_list_field(data, "skills_list")
    proficiency = _get_list_field(data, "proficiency_levels")

    # Map proficiency to level (1-5)
    level_map = {"beginner": 1, "intermediate": 3, "advanced": 4, "expert": 5}

    for i, skill in enumerate(skills):
        if skill:
            level = DEFAULT_RATING  # Default to intermediate
            if i < len(proficiency):
                prof_str = str(proficiency[i]).lower()
                level = level_map.get(prof_str, DEFAULT_RATING)

            items.append({"name": str(skill), "level": level, "category": "technical"})

    return items


def transform_experiences(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted experiences to API format."""
    items = []
    roles = _get_list_field(data, "past_roles")
    companies = _get_list_field(data, "companies")

    for i, role in enumerate(roles):
        if role:
            company = companies[i] if i < len(companies) else "Unknown Company"
            items.append(
                {
                    "title": str(role),
                    "company": str(company),
                    "startDate": data.get("start_date"),
                }
            )

    return items


def transform_certifications(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted certifications to API format."""
    items = []
    certs = _get_list_field(data, "certifications_list", "licenses", "degrees")

    for cert in certs:
        if cert:
            items.append({"name": str(cert), "issuedDate": data.get("issued_date")})

    return items


def transform_current_position(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted current_position to API format."""
    title = data.get("current_title")
    employer = data.get("current_employer")

    if title or employer:
        return [
            {
                "title": str(title or "Current Position"),
                "company": str(employer or "Current Company"),
                "startDate": data.get("start_date"),
                "isCurrent": True,
            }
        ]
    return []


def transform_personality(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted personality_profile to API format (PUT whole context)."""
    traits = _get_list_field(data, "traits_list")

    return {"traits": traits}


def transform_strengths(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted strengths to API format."""
    strengths = _get_list_field(data, "strengths_list")

    return {"strengths": strengths}


def transform_weaknesses(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted weaknesses to API format."""
    weaknesses = _get_list_field(data, "weaknesses_list", "challenges")

    return {"weaknesses": weaknesses}


def transform_motivations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted motivations to API format."""
    items = []
    motivations = _get_list_field(data, "motivations_list", "drivers")

    for motivation in motivations:
        if motivation:
            items.append({"name": str(motivation), "strength": DEFAULT_RATING})

    return items


def transform_work_style(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted work_style to API format."""
    preferences = _get_list_field(data, "work_style_preferences")

    return {"preferences": preferences}


def transform_values(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted values to API format."""
    items = []
    values = _get_list_field(data, "values_list", "priorities")

    for i, value in enumerate(values):
        if value:
            items.append({"name": str(value), "priority": i + 1})

    return items


def transform_knowledge(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted knowledge to API format."""
    areas = _get_list_field(data, "knowledge_areas", "expertise_domains")

    return {"knowledgeAreas": areas}


def transform_learning_velocity(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted learning_velocity to API format."""
    return {"learningSpeed": data.get("learning_speed_indicators", "moderate")}


def transform_preferred_format(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted preferred_format to API format."""
    items = []
    formats = _get_list_field(data, "learning_preferences", "preferred_methods")

    for fmt in formats:
        if fmt:
            items.append({"format": str(fmt), "effectiveness": DEFAULT_RATING})

    return items


def transform_mentors(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted mentors to API format."""
    items = []
    names = _get_list_field(data, "mentor_names")
    roles = _get_list_field(data, "mentor_roles")

    for i, name in enumerate(names):
        if name:
            role = roles[i] if i < len(roles) else "Mentor"
            items.append({"name": str(name), "expertise": str(role), "isActive": True})

    return items


def transform_peers(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted journey_peers to API format."""
    items = []
    peers = _get_list_field(data, "peer_groups", "peer_connections")

    for peer in peers:
        if peer:
            items.append({"name": str(peer), "type": "colleague", "connectionStrength": DEFAULT_RATING})

    return items


def transform_people_helped(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted people_helped to API format (as peers with type=mentee)."""
    items = []
    mentees = _get_list_field(data, "mentees", "people_coached")

    for mentee in mentees:
        if mentee:
            items.append({"name": str(mentee), "type": "mentee", "connectionStrength": DEFAULT_RATING})

    return items


def transform_testimonials(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted testimonials to API format."""
    items = []
    feedback = _get_list_field(data, "feedback_received", "recognition")

    for fb in feedback:
        if fb:
            items.append({"text": str(fb), "source": "colleague"})

    return items


def transform_confidence(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted confidence to API format."""
    level = data.get("confidence_level", "moderate")
    return [{"level": str(level), "timestamp": datetime.now(tz=UTC).isoformat()}]


def transform_stress(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted stress to API format."""
    items = []
    triggers = _get_list_field(data, "stress_factors", "pressure_sources")

    for trigger in triggers:
        if trigger:
            items.append({"trigger": str(trigger), "intensity": DEFAULT_RATING})

    return items


def transform_wellbeing(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted wellbeing to API format."""
    return {"wellbeingState": data.get("wellbeing_state", "good")}


def transform_resilience(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted resilience to API format (as celebrations)."""
    indicators = _get_list_field(data, "resilience_indicators")

    items = []
    for indicator in indicators:
        if indicator:
            items.append({"title": f"Resilience: {indicator}", "type": "personal_growth"})

    return items


# =============================================================================
# Profile Management Functions
# =============================================================================


async def check_profile_exists(user_id: str) -> bool:
    """
    Check if user's Canonical Profile exists.

    Args:
        user_id: User UUID

    Returns:
        True if profile exists, False otherwise
    """
    try:
        result = await _make_request("GET", "/api/harmonia/canonical/profile", user_id=user_id)
        if result and "data" in result:
            logger.info("User profile exists")
            return True
    except HarmoniaAPIError as e:
        logger.warning(f"Profile check failed: {e}")
    return False


async def create_profile(user_id: str, stage: str = "EXPLORATION") -> bool:
    """
    Create a new Canonical Profile for the user.

    Args:
        user_id: User UUID
        stage: Initial journey stage (default: EXPLORATION)

    Returns:
        True if profile created successfully, False otherwise
    """
    data = {"stage": stage}

    try:
        result = await _make_request("POST", "/api/harmonia/canonical/profile", data=data, user_id=user_id)
        if result and "data" in result:
            logger.info(f"Created user profile with stage: {stage}")
            return True
    except HarmoniaAPIError as e:
        logger.error(f"Failed to create user profile: {e}")
    return False


async def ensure_profile_exists(user_id: str) -> bool:
    """
    Ensure user's Canonical Profile exists, creating it if needed.

    Args:
        user_id: User UUID

    Returns:
        True if profile exists or was created, False if creation failed
    """
    if await check_profile_exists(user_id):
        return True

    logger.info("Profile does not exist, creating...")
    return await create_profile(user_id)


# =============================================================================
# Main Storage Function
# =============================================================================


async def store_extracted_information(
    category: str,
    subcategory: str,
    extracted_data: Dict[str, Any],
    user_id: str,
) -> Dict[str, Any]:
    """
    Store extracted information in Canonical Profile (Store A).

    Args:
        category: Primary category (e.g., "professional", "learning")
        subcategory: Subcategory name (e.g., "skills", "dream_roles")
        extracted_data: Extracted information dictionary
        user_id: User UUID

    Returns:
        Dict with "success", "context", "resource", and "created_ids"
    """
    logger.info(f"Storing extracted information: subcategory={subcategory}")

    # Create memory card
    try:
        memory_card = await _make_request(
            "POST",
            "/api/harmonia/journal/memory-cards",
            data={
                "userId": user_id,
                "content": extracted_data.get("content"),
                "type": extracted_data.get("type"),
                "confidence": EXTRACTION_CONFIDENCE,
                "source": {
                    "type": "coach",
                    "sourceId": "pop-ai",
                    "extractedAt": datetime.now(tz=UTC).isoformat(),
                },
                "status": "proposed",
                "linkedContexts": [category, subcategory],
                "createdAt": datetime.now(tz=UTC).isoformat(),
                "validatedAt": None,
            },
            user_id=user_id,
        )
        logger.info(f"Created memory card: {memory_card}")
        return {
            "success": True,
            "context": category,
            "resource": "",
            "created_ids": [memory_card.get("id")],
        }
    except HarmoniaAPIError as e:
        logger.error(f"Failed to create memory card: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Memory card creation via Supabase RPC
# Used by runner_orchestrator (runner flow)
# =============================================================================


def create_memory_proposal_rpc(user_id: str, proposal: dict) -> str | None:
    """
    Write a memory card via the create_memory_proposal RPC function.
    Uses Supabase service client (no user JWT needed).

    Args:
        user_id: User UUID
        proposal: Dict with content, type, confidence, source, rawData, tags,
                  linkedContexts, title, status

    Returns:
        Card UUID string or None on failure
    """
    from src.config import VALID_CARD_TYPES, RPCFunctions, get_supabase

    card_type = proposal.get("type")

    if card_type not in VALID_CARD_TYPES:
        logger.error(f"Invalid card type: {card_type}")
        return None

    try:
        supabase = get_supabase()
        result = supabase.rpc(
            RPCFunctions.CREATE_MEMORY_PROPOSAL,
            {
                "p_user_id": user_id,
                "p_content": proposal["content"],
                "p_type": card_type,
                "p_confidence": proposal["confidence"],
                "p_source": proposal["source"],
                "p_raw_data": proposal.get("rawData"),
                "p_tags": proposal.get("tags", []),
                "p_linked_contexts": proposal.get("linkedContexts", []),
                "p_title": proposal.get("title"),
                "p_status": proposal.get("status", "proposed"),
            },
        ).execute()
        logger.info(f"Created memory card for user {user_id}: type={card_type}")
        return result.data
    except Exception as e:
        if "No ai_training consent" in str(e):
            logger.warning(f"DB consent check blocked card for user {user_id}")
        else:
            logger.error(f"RPC create_memory_proposal failed: {e}")
        return None
