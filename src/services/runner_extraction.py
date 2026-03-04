"""
Runner Extraction — 3-tier pipeline for extracting memory cards from activity completions.

Tier 1: covered_skills → competence cards (deterministic, no LLM)
Tier 2: journey_contributions → mapped cards via rules (deterministic, no LLM)
Tier 3: flat data → LLM extraction (only ~20% of completions)
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from src.config import EXTRACTION_MODEL, QUEUE_POLL_INTERVAL, VALID_CARD_TYPES, get_client_by_provider

logger = logging.getLogger(__name__)

# =============================================================================
# Tier 2: Journey Contributions → Memory Card mapping
# path → (card_type, confidence)
# =============================================================================

JOURNEY_MAPPING = {
    # professional domain
    "professional.profile":                ("experience", 0.90),
    "professional.unique_value_proposition": ("trait", 0.85),
    "professional.skills_positioning":     ("competence", 0.85),
    "professional.positioning_assessment": ("trait", 0.80),
    "professional.development_plan":       ("aspiration", 0.85),
    "professional.development_priorities": ("aspiration", 0.85),
    "professional.career_goals":           ("aspiration", 0.90),
    "professional.mini_bilan":             ("trait", 0.85),
    "professional.market_awareness":       ("competence", 0.80),
    "professional.market_positioning":     ("trait", 0.80),
    "professional.certification":          ("experience", 0.90),
    "professional.validations":            ("competence", 0.85),
    "professional.negotiation":            ("competence", 0.85),
    "professional.objection_handling":     ("competence", 0.85),
    "professional.visibility":             ("competence", 0.80),
    "professional.personal_branding":      ("aspiration", 0.80),
    "professional.montee_competences":     ("competence", 0.85),
    "professional.job_assessments":        ("experience", 0.85),
    "professional.career_clarity":         ("aspiration", 0.85),
    "professional.project":               ("experience", 0.85),
    "professional.activation_readiness":   ("trait", 0.80),
    "professional.research":              ("competence", 0.75),
    "professional.positioning":           ("trait", 0.80),
    "professional.priorities":            ("aspiration", 0.85),
    "professional.kpis":                  ("aspiration", 0.80),
    "professional.patterns":              ("competence", 0.85),

    # personal domain
    "personal.vision_6m":                 ("aspiration", 0.85),
    "personal.goals":                     ("aspiration", 0.85),
    "personal.motivations":               ("trait", 0.85),
    "personal.engagement_baseline":       ("emotion", 0.80),
    "personal.perceived_obstacles":       ("trait", 0.85),
    "personal.action_levers":             ("trait", 0.85),
    "personal.commitment":                ("trait", 0.80),
    "personal.keywords":                  ("trait", 0.75),
    "personal.readiness_signals":         ("aspiration", 0.80),

    # social domain (was network)
    "social":                             ("connection", 0.80),
    "social.circles_mapped":              ("connection", 0.80),
    "social.priority_contacts_selected":  ("connection", 0.80),
    "social.total_contacts_identified":   ("connection", 0.75),

    # psychological domain (was decision + mindset)
    "psychological.validation":           ("trait", 0.80),
    "psychological.market_validation":    ("aspiration", 0.80),
    "psychological.confidence":           ("emotion", 0.80),

    # learning domain (was skills + autonomie)
    "learning.communication":             ("competence", 0.85),
    "learning.interview":                 ("competence", 0.85),
    "learning.autonomie":                 ("competence", 0.80),

    # legacy keys (pre POP-539, still in old completions)
    "network":                            ("connection", 0.80),
    "network.circles_mapped":             ("connection", 0.80),
    "network.priority_contacts_selected": ("connection", 0.80),
    "network.total_contacts_identified":  ("connection", 0.75),
    "market.research":                    ("competence", 0.75),
    "market.positioning":                 ("trait", 0.80),
    "decision.validation":                ("trait", 0.80),
    "decision.market_validation":         ("aspiration", 0.80),
    "skills.communication":               ("competence", 0.85),
    "skills.interview":                   ("competence", 0.85),
    "autonomie":                          ("competence", 0.80),
    "mindset.confidence":                 ("emotion", 0.80),
}

# =============================================================================
# Tier 2: Content templates (French)
# =============================================================================

CONTENT_TEMPLATES = {
    "professional.profile":                "Profil professionnel identifié",
    "professional.unique_value_proposition": "Proposition de valeur unique définie",
    "professional.skills_positioning":     "Positionnement compétences réalisé",
    "professional.positioning_assessment": "Évaluation de positionnement complétée",
    "professional.development_plan":       "Plan de développement à 6 mois défini",
    "professional.development_priorities": "Priorités de développement identifiées",
    "professional.career_goals":           "Objectifs de carrière définis",
    "professional.mini_bilan":             "Mini-bilan express réalisé",
    "professional.market_awareness":       "Connaissance du marché évaluée",
    "professional.market_positioning":     "Positionnement marché analysé",
    "professional.certification":          "Certification: scénario complété",
    "professional.validations":            "Évidences professionnelles validées",
    "professional.negotiation":            "Préparation à la négociation réalisée",
    "professional.objection_handling":     "Gestion des objections préparée",
    "professional.visibility":             "Audit de visibilité complété",
    "professional.personal_branding":      "Stratégie de personal branding définie",
    "professional.montee_competences":     "Plan de montée en compétences établi",
    "professional.job_assessments":        "Évaluations de postes réalisées",
    "professional.career_clarity":         "Clarté de trajectoire professionnelle renforcée",
    "professional.project":               "Projet professionnel évalué",
    "professional.activation_readiness":   "Préparation à l'activation évaluée",
    "professional.research":              "Recherche marché réalisée",
    "professional.positioning":           "Positionnement marché ajusté",
    "professional.priorities":            "Priorités professionnelles définies",
    "professional.kpis":                  "KPIs professionnels définis",
    "professional.patterns":              "Patterns de compétences identifiés",
    "personal.vision_6m":                 "Vision à 6 mois définie",
    "personal.goals":                     "Objectifs personnels identifiés",
    "personal.motivations":               "Motivations identifiées",
    "personal.engagement_baseline":       "Baseline d'engagement mesurée",
    "personal.perceived_obstacles":       "Obstacles perçus identifiés",
    "personal.action_levers":             "Leviers d'action identifiés",
    "personal.commitment":                "Engagement formalisé",
    "personal.keywords":                  "Mots-clés personnels identifiés",
    "personal.readiness_signals":         "Signaux de préparation identifiés",
    "social":                             "Réseau professionnel cartographié",
    "social.circles_mapped":              "Cercles de réseau cartographiés",
    "social.priority_contacts_selected":  "Contacts prioritaires sélectionnés",
    "social.total_contacts_identified":   "Contacts totaux identifiés",
    "psychological.validation":           "Validation de décision complétée",
    "psychological.market_validation":    "Validation marché réalisée",
    "psychological.confidence":           "Niveau de confiance évalué",
    "learning.communication":             "Compétences de communication évaluées",
    "learning.interview":                 "Préparation aux entretiens réalisée",
    "learning.autonomie":                 "Niveau d'autonomie évalué",
    # legacy keys
    "network":                            "Réseau professionnel cartographié",
    "network.circles_mapped":             "Cercles de réseau cartographiés",
    "network.priority_contacts_selected": "Contacts prioritaires sélectionnés",
    "network.total_contacts_identified":  "Contacts totaux identifiés",
    "market.research":                    "Recherche marché réalisée",
    "market.positioning":                 "Positionnement marché analysé",
    "decision.validation":                "Validation de décision complétée",
    "decision.market_validation":         "Validation marché réalisée",
    "skills.communication":               "Compétences de communication évaluées",
    "skills.interview":                   "Préparation aux entretiens réalisée",
    "autonomie":                          "Niveau d'autonomie évalué",
    "mindset.confidence":                 "Niveau de confiance évalué",
}


# =============================================================================
# Helper functions
# =============================================================================

def _resolve_nested_path(data: dict, path: str):
    """Resolve 'a.b.c' into data['a']['b']['c']."""
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _generate_content(path: str, value) -> str:
    """Generate JSON content matching the chat flow tree-view format.

    The frontend renders content as a tree view when it's valid JSON
    wrapped as {subcategory: fields}. Arrays of strings are joined
    into readable comma-separated values. The CONTENT_TEMPLATES label
    is used as key for better readability.
    """
    label = CONTENT_TEMPLATES.get(
        path, path.replace(".", " > ").replace("_", " ").title()
    )

    # Arrays of strings → join into readable comma-separated text
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        value = ", ".join(value)

    return json.dumps({label: value}, ensure_ascii=False, default=str)


def _build_source(completion: dict) -> dict:
    """Build POP-507 source object from completion record."""
    responses = completion.get("responses") or {}
    metadata = responses.get("_runner_metadata") or {}
    return {
        "type": "runner",
        "sourceId": metadata.get("runner_type", "unknown"),
        "activityId": completion.get("activity_id"),
        "extractedAt": datetime.now(UTC).isoformat(),
    }


# =============================================================================
# Tier 1: covered_skills → competence cards
# =============================================================================

def _extract_tier1(responses: dict, source: dict) -> list[dict]:
    """Tier 1: deterministic extraction from covered_skills (RNCP codes)."""
    covered_skills = responses.get("covered_skills", [])
    if not covered_skills or not isinstance(covered_skills, list):
        return []

    proposals = []
    for skill in covered_skills:
        if not isinstance(skill, dict) or "code" not in skill:
            continue
        code = skill["code"]
        level = skill.get("level", 1)
        proposals.append({
            "content": json.dumps({f"Compétence RNCP {code}": {"code": code, "niveau": level}}, ensure_ascii=False),
            "type": "competence",
            "confidence": 0.95,
            "source": source,
            "rawData": skill,
            "tags": ["rncp", code],
            "linkedContexts": ["learning", "knowledge_and_credentials"],
            "title": f"Compétence RNCP {code}",
            "status": "proposed",
        })

    if proposals:
        logger.info(f"Tier 1: extracted {len(proposals)} competence cards from covered_skills")
    return proposals


# =============================================================================
# Tier 2: journey_contributions → mapped cards
# =============================================================================

def _extract_tier2(responses: dict, source: dict) -> list[dict]:
    """Tier 2: deterministic extraction from journey_contributions via JOURNEY_MAPPING."""
    journey = responses.get("journey_contributions")
    if not journey or not isinstance(journey, dict):
        return []

    proposals = []
    for path, (card_type, confidence) in JOURNEY_MAPPING.items():
        value = _resolve_nested_path(journey, path)
        if value is None or value == {} or value == []:
            continue
        content = _generate_content(path, value)
        parts = path.split(".", 1)
        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else category
        title = CONTENT_TEMPLATES.get(path, subcategory.replace("_", " ").title())
        proposals.append({
            "content": content,
            "type": card_type,
            "confidence": confidence,
            "source": source,
            "rawData": {"path": path, "value": value},
            "tags": [category],
            "linkedContexts": [category, subcategory],
            "title": title,
            "status": "proposed",
        })

    if proposals:
        logger.info(f"Tier 2: extracted {len(proposals)} cards from journey_contributions")
    return proposals


# =============================================================================
# Tier 3: flat data → LLM extraction
# =============================================================================

_SKIP_KEYS = {
    "_runner_metadata", "_metadata", "_ai_context", "_activity_context",
    "covered_skills", "journey_contributions",
    "dashboard_data", "dashboard_summary",
    "completed_at", "duration_ms", "credits_consumed",
    "version",
}

LLM_SYSTEM_PROMPT = """You are a memory card extractor for a career coaching platform.
Extract memory card proposals from this runner completion data.

Each card must have:
- subcategory: string (one of the subcategories below)
- fields: object with structured data (see schema per subcategory)
- type: string (the card type for that subcategory)
- confidence: number (0.70-0.80)
- tags: string[] (1-3 relevant tags in French)

Subcategories and their schemas:

work_history (type=experience): {role, company, isCurrent, startDate, endDate, responsibilities, achievements}
professional_aspirations (type=aspiration): {dreamRole, targetCompany, targetIndustry, targetTimeframe, skillGapsToAddress}
professional_achievements (type=experience): {type, title, organization, date, description}
knowledge_and_credentials (type=competence): {type, name, level, yearsExperience, institution}
languages (type=competence): {language, proficiency, certification}
learning_agenda (type=aspiration): {gapOrGoal, description, targetDate, preferredFormat}
network_and_networking (type=connection): {type, name, role, organization, engagementLevel, networkingGoal}
mentorship (type=connection): {direction, name, role, organization, guidanceAreas}
emotional_state (type=emotion): {dimension, context, intensity, duration}
mindset_and_values (type=emotion): {category, value, strength, description}

Return a JSON object: {"cards": [...]}
Each card: {"subcategory": "...", "fields": {...}, "type": "...", "confidence": 0.75, "tags": [...]}
Only fill fields that have actual data. Omit fields with no data.
If no meaningful data can be extracted, return {"cards": []}.
Do NOT extract dashboard layout, metadata, or technical fields."""


async def _extract_tier3(responses: dict, source: dict) -> list[dict]:
    """Tier 3: LLM-based extraction from remaining flat data."""
    remaining = {
        k: v for k, v in responses.items()
        if k not in _SKIP_KEYS and v is not None and v != {} and v != []
    }

    if not remaining:
        return []

    # Include _ai_context if available for better extraction
    ai_context = responses.get("_ai_context")
    user_prompt = f"Runner completion data:\n{json.dumps(remaining, ensure_ascii=False, default=str)}"
    if ai_context:
        user_prompt += f"\n\nSemantic annotations:\n{json.dumps(ai_context, ensure_ascii=False, default=str)}"

    try:
        client = get_client_by_provider("openai")
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw = response.choices[0].message.content
        logger.info(f"Tier 3: LLM raw response: {raw[:500]}")
        parsed = json.loads(raw)

        # Handle {"cards": [...]}, {"proposals": [...]}, {"memoryCards": [...]}, or [...]
        if isinstance(parsed, list):
            items = parsed
        else:
            # Find the first list value in the response dict
            items = next((v for v in parsed.values() if isinstance(v, list)), [])
        if not isinstance(items, list):
            items = []

        proposals = []
        for item in items:
            if not isinstance(item, dict):
                continue
            card_type = item.get("type")
            if card_type not in VALID_CARD_TYPES:
                continue
            subcategory = item.get("subcategory", card_type)
            fields = item.get("fields", {})
            if not isinstance(fields, dict) or not fields:
                continue
            # Tree-view JSON: {subcategory: {field: value, ...}} — same as chat flow
            content_tree = json.dumps({subcategory: fields}, ensure_ascii=False)
            title = subcategory.replace("_", " ").title()
            proposals.append({
                "content": content_tree,
                "type": card_type,
                "confidence": min(max(float(item.get("confidence", 0.75)), 0.0), 1.0),
                "source": source,
                "rawData": {"llm_extracted": True, "original_keys": list(remaining.keys())},
                "tags": item.get("tags", [])[:3],
                "linkedContexts": item.get("linkedContexts", []),
                "title": title,
                "status": "proposed",
            })

        if proposals:
            logger.info(f"Tier 3: LLM extracted {len(proposals)} cards")

        # Rate limiting: avoid hitting OpenAI RPM limits
        await asyncio.sleep(QUEUE_POLL_INTERVAL / 10)

        return proposals

    except Exception as e:
        logger.error(f"Tier 3 LLM extraction failed: {e}")
        return []


# =============================================================================
# Main entry point
# =============================================================================

async def process_completion(completion: dict) -> list[dict]:
    """
    Process a single activity completion through the 3-tier extraction pipeline.

    Args:
        completion: Full activity_completions row (dict with 'responses', 'activity_id', etc.)

    Returns:
        List of memory card proposals ready for create_memory_proposal_rpc()
    """
    responses = completion.get("responses") or {}
    source = _build_source(completion)

    # Tier 1: covered_skills (deterministic)
    tier1 = _extract_tier1(responses, source)

    # Tier 2: journey_contributions (deterministic)
    tier2 = _extract_tier2(responses, source)

    proposals = tier1 + tier2

    # Tier 3: LLM extraction (only if Tiers 1-2 didn't cover the data)
    tier3 = []
    if not proposals or not tier2:
        tier3 = await _extract_tier3(responses, source)
        proposals.extend(tier3)

    # Validate all proposals
    valid = [
        p for p in proposals
        if p.get("type") in VALID_CARD_TYPES
        and p.get("content")
        and isinstance(p.get("confidence"), (int, float))
        and 0 <= p["confidence"] <= 1
    ]
    dropped = len(proposals) - len(valid)
    if dropped:
        logger.warning(f"Dropped {dropped} invalid proposals")

    logger.info(
        f"Extraction complete for completion {completion.get('id', '?')}: "
        f"{len(valid)} valid proposals (T1={len(tier1)}, T2={len(tier2)}, T3={len(tier3)})"
    )

    return valid
