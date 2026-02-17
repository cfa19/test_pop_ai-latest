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

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import requests

from src.config import NEXT_PUBLIC_BASE_URL, Tables

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Base path for Canonical Profile API
CANONICAL_BASE = "/api/harmonia/canonical/contexts"

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

def _make_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    user_token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Make HTTP request to Harmonia API.

    Args:
        method: HTTP method (GET, POST, PATCH, DELETE, PUT)
        endpoint: API endpoint path
        data: Request body data
        headers: Additional headers
        user_token: User JWT token (optional, uses service key if not provided)

    Returns:
        Response JSON or None if error
    """
    if not NEXT_PUBLIC_BASE_URL:
        logger.error("Harmonia API configuration missing (SUPABASE_URL or SUPABASE_KEY)")
        return None

    url = f"{NEXT_PUBLIC_BASE_URL}{endpoint}"

    # Build headers
    request_headers = {
        # "apikey": HARMONIA_API_KEY,s
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
        if hasattr(e, 'response') and e.response is not None:
            try:
                logger.error(f"Response body: {e.response.text}")
            except Exception:
                pass
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling Harmonia API: {e}")
        return None


# =============================================================================
# Data Transform Functions
# =============================================================================

def transform_dream_roles(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted dream_roles to API format."""
    items = []

    # Extract occupations/job titles
    titles = data.get("occupations", []) or data.get("job_titles", []) or []
    if isinstance(titles, str):
        titles = [t.strip() for t in titles.split(",")]

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
    goals = data.get("life_goals", []) or data.get("lifestyle_aspirations", []) or []

    if isinstance(goals, str):
        goals = [g.strip() for g in goals.split(",")]

    for goal in goals:
        if goal:
            items.append({"title": str(goal), "progress": 0})

    return items


def transform_impact_legacy(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted impact_legacy to API format (as life goals)."""
    items = []
    impacts = data.get("impact_goals", []) or data.get("legacy_aspirations", []) or []

    if isinstance(impacts, str):
        impacts = [i.strip() for i in impacts.split(",")]

    for impact in impacts:
        if impact:
            items.append({"title": f"Impact: {impact}", "progress": 0})

    return items


def transform_skills(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted skills to API format."""
    items = []
    skills = data.get("skills_list", []) or []
    proficiency = data.get("proficiency_levels", []) or []

    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",")]
    if isinstance(proficiency, str):
        proficiency = [p.strip() for p in proficiency.split(",")]

    # Map proficiency to level (1-5)
    level_map = {"beginner": 1, "intermediate": 3, "advanced": 4, "expert": 5}

    for i, skill in enumerate(skills):
        if skill:
            level = 3  # Default to intermediate
            if i < len(proficiency):
                prof_str = str(proficiency[i]).lower()
                level = level_map.get(prof_str, 3)

            items.append({"name": str(skill), "level": level, "category": "technical"})

    return items


def transform_experiences(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted experiences to API format."""
    items = []
    roles = data.get("past_roles", []) or []
    companies = data.get("companies", []) or []

    if isinstance(roles, str):
        roles = [r.strip() for r in roles.split(",")]
    if isinstance(companies, str):
        companies = [c.strip() for c in companies.split(",")]

    for i, role in enumerate(roles):
        if role:
            company = companies[i] if i < len(companies) else "Unknown Company"
            items.append({
                "title": str(role),
                "company": str(company),
                "startDate": datetime.now().strftime("%Y-%m-%d")  # Placeholder
            })

    return items


def transform_certifications(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted certifications to API format."""
    items = []
    certs = (data.get("certifications_list", []) or
             data.get("licenses", []) or
             data.get("degrees", []) or [])

    if isinstance(certs, str):
        certs = [c.strip() for c in certs.split(",")]

    for cert in certs:
        if cert:
            items.append({"name": str(cert), "issuedDate": datetime.now().strftime("%Y-%m-%d")})

    return items


def transform_current_position(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted current_position to API format."""
    title = data.get("current_title")
    employer = data.get("current_employer")

    if title or employer:
        return [{
            "title": str(title or "Current Position"),
            "company": str(employer or "Current Company"),
            "startDate": datetime.now().strftime("%Y-%m-%d"),
            "isCurrent": True
        }]
    return []


def transform_personality(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted personality_profile to API format (PUT whole context)."""
    traits = data.get("traits_list", [])
    if isinstance(traits, str):
        traits = [t.strip() for t in traits.split(",")]

    return {"traits": traits}


def transform_strengths(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted strengths to API format."""
    strengths = data.get("strengths_list", [])
    if isinstance(strengths, str):
        strengths = [s.strip() for s in strengths.split(",")]

    return {"strengths": strengths}


def transform_weaknesses(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted weaknesses to API format."""
    weaknesses = data.get("weaknesses_list", []) or data.get("challenges", [])
    if isinstance(weaknesses, str):
        weaknesses = [w.strip() for w in weaknesses.split(",")]

    return {"weaknesses": weaknesses}


def transform_motivations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted motivations to API format."""
    items = []
    motivations = data.get("motivations_list", []) or data.get("drivers", []) or []

    if isinstance(motivations, str):
        motivations = [m.strip() for m in motivations.split(",")]

    for motivation in motivations:
        if motivation:
            items.append({"name": str(motivation), "strength": 3})

    return items


def transform_work_style(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted work_style to API format."""
    preferences = data.get("work_style_preferences", [])
    if isinstance(preferences, str):
        preferences = [p.strip() for p in preferences.split(",")]

    return {"preferences": preferences}


def transform_values(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted values to API format."""
    items = []
    values = data.get("values_list", []) or data.get("priorities", []) or []

    if isinstance(values, str):
        values = [v.strip() for v in values.split(",")]

    for i, value in enumerate(values):
        if value:
            items.append({"name": str(value), "priority": i + 1})

    return items


def transform_knowledge(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted knowledge to API format."""
    areas = data.get("knowledge_areas", []) or data.get("expertise_domains", []) or []
    if isinstance(areas, str):
        areas = [a.strip() for a in areas.split(",")]

    return {"knowledgeAreas": areas}


def transform_learning_velocity(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted learning_velocity to API format."""
    return {"learningSpeed": data.get("learning_speed_indicators", "moderate")}


def transform_preferred_format(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted preferred_format to API format."""
    items = []
    formats = data.get("learning_preferences", []) or data.get("preferred_methods", []) or []

    if isinstance(formats, str):
        formats = [f.strip() for f in formats.split(",")]

    for fmt in formats:
        if fmt:
            items.append({"format": str(fmt), "effectiveness": 3})

    return items


def transform_mentors(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted mentors to API format."""
    items = []
    names = data.get("mentor_names", []) or []
    roles = data.get("mentor_roles", []) or []

    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    if isinstance(roles, str):
        roles = [r.strip() for r in roles.split(",")]

    for i, name in enumerate(names):
        if name:
            role = roles[i] if i < len(roles) else "Mentor"
            items.append({"name": str(name), "expertise": str(role), "isActive": True})

    return items


def transform_peers(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted journey_peers to API format."""
    items = []
    peers = data.get("peer_groups", []) or data.get("peer_connections", []) or []

    if isinstance(peers, str):
        peers = [p.strip() for p in peers.split(",")]

    for peer in peers:
        if peer:
            items.append({"name": str(peer), "type": "colleague", "connectionStrength": 3})

    return items


def transform_people_helped(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted people_helped to API format (as peers with type=mentee)."""
    items = []
    mentees = data.get("mentees", []) or data.get("people_coached", []) or []

    if isinstance(mentees, str):
        mentees = [m.strip() for m in mentees.split(",")]

    for mentee in mentees:
        if mentee:
            items.append({"name": str(mentee), "type": "mentee", "connectionStrength": 3})

    return items


def transform_testimonials(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted testimonials to API format."""
    items = []
    feedback = data.get("feedback_received", []) or data.get("recognition", []) or []

    if isinstance(feedback, str):
        feedback = [f.strip() for f in feedback.split(",")]

    for fb in feedback:
        if fb:
            items.append({"text": str(fb), "source": "colleague"})

    return items


def transform_confidence(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted confidence to API format."""
    level = data.get("confidence_level", "moderate")
    return [{"level": str(level), "timestamp": datetime.now().isoformat()}]


def transform_stress(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted stress to API format."""
    items = []
    triggers = data.get("stress_factors", []) or data.get("pressure_sources", []) or []

    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",")]

    for trigger in triggers:
        if trigger:
            items.append({"trigger": str(trigger), "intensity": 3})

    return items


def transform_wellbeing(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted wellbeing to API format."""
    return {"wellbeingState": data.get("wellbeing_state", "good")}


def transform_resilience(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform extracted resilience to API format (as celebrations)."""
    indicators = data.get("resilience_indicators", [])
    if isinstance(indicators, str):
        indicators = [i.strip() for i in indicators.split(",")]

    items = []
    for indicator in indicators:
        if indicator:
            items.append({"title": f"Resilience: {indicator}", "type": "personal_growth"})

    return items


# =============================================================================
# Profile Management Functions
# =============================================================================

def check_profile_exists(user_token: str) -> bool:
    """
    Check if user's Canonical Profile exists.

    Args:
        user_token: User JWT token

    Returns:
        True if profile exists, False otherwise
    """
    result = _make_request(
        "GET",
        "/api/harmonia/canonical/profile",
        user_token=user_token
    )

    # If we get a successful response with data, profile exists
    if result and "data" in result:
        logger.info("✓ User profile exists")
        return True

    logger.info("User profile does not exist")
    return False


def create_profile(user_token: str, stage: str = "EXPLORATION") -> bool:
    """
    Create a new Canonical Profile for the user.

    Args:
        user_token: User JWT token
        stage: Initial journey stage (default: EXPLORATION)

    Returns:
        True if profile created successfully, False otherwise
    """
    data = {"stage": stage}

    result = _make_request(
        "POST",
        "/api/harmonia/canonical/profile",
        data=data,
        user_token=user_token
    )

    if result and "data" in result:
        logger.info(f"✓ Created user profile with stage: {stage}")
        return True

    logger.error("Failed to create user profile")
    return False


def ensure_profile_exists(user_token: str) -> bool:
    """
    Ensure user's Canonical Profile exists, creating it if needed.

    Args:
        user_token: User JWT token

    Returns:
        True if profile exists or was created, False if creation failed
    """
    if check_profile_exists(user_token):
        return True

    logger.info("Profile does not exist, creating...")
    return create_profile(user_token)


# =============================================================================
# Entity → Card Type mapping
# =============================================================================
# Maps entity/sub_entity names to memory card types used in the DB.
# Valid types (from real data): competence, experience, preference, aspiration,
#                               trait, emotion, connection

_ENTITY_TO_CARD_TYPE: Dict[str, str] = {
    # professional entities
    "current_position": "competence",
    "professional_experience": "competence",
    "awards": "competence",
    "licenses_and_permits": "competence",
    "professional_aspirations": "aspiration",
    "volunteer_experience": "competence",
    # learning entities
    "current_skills": "competence",
    "languages": "competence",
    "education_history": "competence",
    "learning_gaps": "aspiration",
    "learning_aspirations": "aspiration",
    "certifications": "competence",
    "knowledge_areas": "competence",
    "learning_preferences": "preference",
    "learning_history": "competence",
    "publications": "competence",
    "academic_awards": "competence",
    # social entities
    "mentors": "connection",
    "mentees": "connection",
    "professional_network": "connection",
    "recommendations": "connection",
    "networking": "connection",
    # psychological entities
    "personality_profile": "trait",
    "values": "preference",
    "motivations": "preference",
    "working_style_preferences": "preference",
    "confidence_and_self_perception": "emotion",
    "career_decision_making_style": "trait",
    "work_environment_preferences": "preference",
    "stress_and_coping": "emotion",
    "emotional_intelligence": "emotion",
    "growth_mindset": "trait",
    # personal entities
    "personal_life": "trait",
    "health_and_wellbeing": "emotion",
    "living_situation": "trait",
    "financial_situation": "trait",
    "personal_goals": "aspiration",
    "personal_projects": "competence",
    "lifestyle_preferences": "preference",
    "life_constraints": "trait",
    "life_enablers": "trait",
    "major_life_events": "emotion",
    "personal_values": "preference",
    "life_satisfaction": "emotion",
    # sub-entity level (fallback)
    "dream_roles": "aspiration",
    "compensation_expectations": "aspiration",
    "skills": "competence",
    "stress": "emotion",
    "confidence": "emotion",
    "strengths": "trait",
    "weaknesses": "trait",
    "life_goals": "aspiration",
    "impact_legacy": "aspiration",
}


def _resolve_card_type(entity: Optional[str], sub_entity: str) -> str:
    """Resolve the memory card type from entity or sub_entity name."""
    if entity and entity in _ENTITY_TO_CARD_TYPE:
        return _ENTITY_TO_CARD_TYPE[entity]
    if sub_entity in _ENTITY_TO_CARD_TYPE:
        return _ENTITY_TO_CARD_TYPE[sub_entity]
    return "competence"  # safe default (most common in real data)


# =============================================================================
# Sub-entity → primary rawData field name (from taxonomy)
# =============================================================================
# When the LLM returns a simple value (not a dict), we wrap it with
# the correct field name from the taxonomy instead of a generic "value".

_SUB_ENTITY_PRIMARY_FIELD: Dict[str, str] = {
    # professional
    "role": "role", "company": "company", "compensation": "compensation",
    "dream_roles": "desired_roles", "compensation_expectations": "target_salary",
    "desired_work_environment": "work_mode", "career_change_considerations": "considering_change",
    "job_search_status": "currently_searching", "awards": "awards",
    "volunteer_experience": "volunteer_roles", "past_roles": "past_roles",
    # learning
    "skills": "skill_name", "languages": "language", "degrees": "degrees",
    "education_history": "degrees", "skill_gaps": "missing_skills",
    "knowledge_gaps": "missing_knowledge", "skill_aspirations": "target_skills",
    "education_aspirations": "desired_degrees", "certification_aspirations": "target_certs",
    "certifications": "earned_certs", "knowledge_areas": "expertise_domains",
    "learning_preferences": "preferred_formats", "learning_history": "past_courses",
    "publications": "publications", "academic_awards": "academic_awards",
    "experience": "years_experience", "proficiency": "proficiency",
    # social
    "mentors": "mentor_name", "mentees": "mentee_name",
    "professional_network": "connections", "recommendations": "testimonial_from",
    "networking_activities": "activity_type", "networking_goals": "target_connections",
    "networking_preferences": "preferred_formats",
    # psychological
    "personality_profile": "personality_type", "values": "professional_values",
    "motivations": "intrinsic_motivations", "working_style_preferences": "work_style",
    "confidence_levels": "overall_confidence", "confidence_and_self_perception": "overall_confidence",
    "imposter_syndrome_and_doubt": "imposter_level",
    "self_talk_and_validation": "inner_critic_strength",
    "confidence_building_strategies": "strategies_that_help",
    "career_decision_making_style": "decision_style",
    "work_environment_preferences": "ideal_environment",
    "stress_and_coping": "stress_level", "emotional_intelligence": "self_awareness",
    "growth_mindset": "mindset_level",
    # personal
    "personal_life": "life_stage", "physical_health": "overall_health",
    "mental_health": "conditions", "addictions_or_recovery": "addiction_type",
    "overall_wellbeing": "wellbeing_score", "living_situation": "location",
    "financial_situation": "stability", "personal_goals": "non_career_goals",
    "personal_projects": "project_name", "lifestyle_preferences": "work_life_balance",
    "life_constraints": "constraint_type", "life_enablers": "enabler_type",
    "major_life_events": "event_type", "personal_values": "life_values",
    "life_satisfaction": "overall_satisfaction",
    # common fallbacks
    "stress": "stress_level", "confidence": "confidence_level",
    "strengths": "strengths", "weaknesses": "areas_for_growth",
    "life_goals": "non_career_goals", "impact_legacy": "life_goals",
}


def _build_content(sub_entity: str, raw_data: Dict[str, Any]) -> str:
    """Build content as nested JSON so the frontend renders a collapsible tree branch.

    Wraps data under the sub-entity key → JsonTreeView shows a collapsible node.
    Strips null/empty values so only meaningful data is displayed.
    """
    if not raw_data:
        return json.dumps({sub_entity: None}, ensure_ascii=False)

    # Strip nulls/empty for cleaner display
    clean = {k: v for k, v in raw_data.items() if v is not None and v != "" and v != []}
    if not clean:
        return json.dumps({sub_entity: None}, ensure_ascii=False)

    # Wrap under sub-entity key → creates a collapsible branch in JsonTreeView
    return json.dumps({sub_entity: clean}, ensure_ascii=False)


def _build_raw_data(sub_entity: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build rawData JSONB using taxonomy field names.

    When the LLM returns a dict, use those fields directly.
    When it returns a simple value, wrap it with the taxonomy field name.
    """
    data = {k: v for k, v in extracted_data.items()
            if k not in ("content", "type")}

    # If only "value" key exists, rename it to the taxonomy field name
    if list(data.keys()) == ["value"]:
        field_name = _SUB_ENTITY_PRIMARY_FIELD.get(sub_entity, sub_entity)
        return {field_name: data["value"]}

    # Dict with proper field names from LLM — use as-is
    return data


# =============================================================================
# Main Storage Function
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
    Store extracted information as a memory card directly in Supabase (Store B).

    Inserts into the `memory_cards` table following POP-507 v2 schema.

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

    # Resolve card type from entity/sub_entity (must match DB CHECK constraint)
    card_type = _resolve_card_type(entity, subcategory)

    # Structured extraction data — taxonomy field names (v2 guide)
    raw_data = _build_raw_data(subcategory, extracted_data)

    # Content as JSON string — frontend renders as collapsible tree
    content = _build_content(subcategory, raw_data)

    # Source provenance (JSONB) — no sessionId per v2 guide
    source = {
        "type": "coach",
        "sourceId": conversation_id or "unknown",
        "extractedAt": now,
    }

    try:
        result = (
            supabase.table(Tables.MEMORY_CARDS)
            .insert({
                "id": card_id,
                "user_id": user_id,
                "content": content,
                "type": card_type,
                "confidence": round(min(confidence, 1.0), 2),
                "source": source,
                "status": "proposed",
                "tags": [category, entity or subcategory, subcategory],
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
