"""
Runner Extraction — 3-tier pipeline for extracting memory cards from activity completions.

Tier 1: _ai_context.data_semantics → deterministic extraction using canonical_target mapping
Tier 2: journey_contributions → mapped cards via JOURNEY_MAPPING rules (deterministic)
Tier 3: LLM fallback → uses EXTRACTION_SCHEMAS from chat flow for remaining data
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from src.config import EXTRACTION_MODEL, QUEUE_POLL_INTERVAL, VALID_CARD_TYPES, get_client_by_provider
from src.schemas import EXTRACTION_SCHEMAS

logger = logging.getLogger(__name__)

# =============================================================================
# canonical_target → (subcategory_key, card_type)
# Maps _ai_context canonical targets to the extraction schema keys
# =============================================================================

_CANONICAL_TO_SCHEMA = {}
for schema_key, schema in EXTRACTION_SCHEMAS.items():
    _CANONICAL_TO_SCHEMA[schema_key] = (schema_key, schema["type"])

# Additional mappings for canonical_target formats like "professional.experience"
_CANONICAL_TARGET_ALIASES = {
    "professional.experience": ("work_history", "experience"),
    "professional.aspirations": ("professional_aspirations", "aspiration"),
    "professional.achievements": ("professional_achievements", "experience"),
    "professional.strengths": ("work_history", "experience"),
    "professional.development_priorities": ("professional_aspirations", "aspiration"),
    "professional.development_plan": ("professional_aspirations", "aspiration"),
    "learning.knowledge_and_credentials": ("knowledge_and_credentials", "competence"),
    "learning.languages": ("languages", "competence"),
    "learning.educationHistory": ("knowledge_and_credentials", "competence"),
    "personal.personalityProfile": ("mindset_and_values", "emotion"),
    "personal.personalTraits": ("mindset_and_values", "emotion"),
    "social.network": ("network_and_networking", "connection"),
}


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
    "personal.identityKeywords":          ("trait", 0.80),
    # social domain
    "social":                             ("connection", 0.80),
    "social.circles_mapped":              ("connection", 0.80),
    "social.priority_contacts_selected":  ("connection", 0.80),
    "social.total_contacts_identified":   ("connection", 0.75),
    # psychological domain
    "psychological.validation":           ("trait", 0.80),
    "psychological.market_validation":    ("aspiration", 0.80),
    "psychological.confidence":           ("emotion", 0.80),
    # learning domain
    "learning.communication":             ("competence", 0.85),
    "learning.interview":                 ("competence", 0.85),
    "learning.autonomie":                 ("competence", 0.80),
}

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
    "personal.identityKeywords":          "Mots-clés d'identité identifiés",
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


def _build_source(completion: dict) -> dict:
    """Build source object from completion record."""
    responses = completion.get("responses") or {}
    metadata = responses.get("_runner_metadata") or {}
    return {
        "type": "runner",
        "sourceId": metadata.get("runner_type", "unknown"),
        "activityId": completion.get("activity_id"),
        "extractedAt": datetime.now(UTC).isoformat(),
    }


def _make_tree_content(subcategory: str, value) -> str:
    """Wrap value in tree-view JSON format: {subcategory: value}."""
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        value = ", ".join(value)
    return json.dumps({subcategory: value}, ensure_ascii=False, default=str)


# =============================================================================
# Tier 1: _ai_context.data_semantics → direct extraction (no LLM)
# =============================================================================

def _extract_tier1(responses: dict, source: dict) -> tuple[list[dict], set[str]]:
    """Tier 1: deterministic extraction using _ai_context.data_semantics.

    Returns:
        (proposals, extracted_paths) — extracted_paths are data paths already covered
    """
    ai_context = responses.get("_ai_context")
    if not ai_context or not isinstance(ai_context, dict):
        return [], set()

    semantics = ai_context.get("data_semantics")
    if not semantics or not isinstance(semantics, dict):
        return [], set()

    proposals = []
    extracted_paths = set()

    for data_path, meta in semantics.items():
        if not isinstance(meta, dict):
            continue

        canonical = meta.get("canonical_target", "")
        if not canonical:
            continue

        # Resolve the data path to get the actual value
        value = _resolve_nested_path(responses, data_path)
        if value is None or value == {} or value == []:
            continue

        # Find the schema key and card type from canonical target
        schema_key, card_type = _CANONICAL_TARGET_ALIASES.get(
            canonical,
            _CANONICAL_TO_SCHEMA.get(canonical.split(".")[-1], (None, None))
        )
        if not schema_key or card_type not in VALID_CARD_TYPES:
            continue

        # Handle lists (e.g., cvData.experiences is a list of jobs)
        items_to_process = value if isinstance(value, list) else [value]

        for item in items_to_process:
            if isinstance(item, dict):
                # Filter out empty/null fields
                clean = {k: v for k, v in item.items()
                         if v is not None and v != "" and v != "null"
                         and k not in ("id", "metadata", "_ai_label")}
                if not clean:
                    continue
                content = json.dumps({schema_key: clean}, ensure_ascii=False)
                title = schema_key.replace("_", " ").title()
            elif isinstance(item, str) and item.strip():
                content = json.dumps({schema_key: item}, ensure_ascii=False)
                title = schema_key.replace("_", " ").title()
            else:
                continue

            proposals.append({
                "content": content,
                "type": card_type,
                "confidence": 0.85,
                "source": source,
                "rawData": {"data_path": data_path, "canonical_target": canonical},
                "tags": [canonical.split(".")[0]] if "." in canonical else [],
                "linkedContexts": canonical.split(".") if "." in canonical else [],
                "title": title,
                "status": "proposed",
            })

        extracted_paths.add(data_path)

    if proposals:
        logger.info(f"Tier 1: extracted {len(proposals)} cards from _ai_context.data_semantics")
    return proposals, extracted_paths


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

        label = CONTENT_TEMPLATES.get(path, path.replace(".", " > ").replace("_", " ").title())
        content = _make_tree_content(label, value)

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
# Tier 3: LLM fallback — uses same EXTRACTION_SCHEMAS as chat flow
# =============================================================================

_SKIP_KEYS = {
    "_runner_metadata", "_metadata", "_ai_context", "_activity_context",
    "covered_skills", "journey_contributions",
    "dashboard_data", "dashboard_summary",
    "completed_at", "duration_ms", "credits_consumed",
    "version", "report_html",
}


def _build_llm_prompt() -> str:
    """Build LLM system prompt from EXTRACTION_SCHEMAS (same schemas as chat)."""
    schema_lines = []
    for key, schema in EXTRACTION_SCHEMAS.items():
        fields = ", ".join(schema["fields"])
        schema_lines.append(f"{key} (type={schema['type']}): {{{fields}}}")

    return f"""You are a memory card extractor for a career coaching platform.
Extract memory card proposals from this runner completion data.

Each card must have:
- subcategory: string (one of the subcategories below)
- fields: object with structured data (see schema per subcategory)
- type: string (the card type for that subcategory)
- confidence: number (0.70-0.80)
- tags: string[] (1-3 relevant tags in French)

Subcategories and their schemas:

{chr(10).join(schema_lines)}

Return a JSON object: {{"cards": [...]}}
Each card: {{"subcategory": "...", "fields": {{...}}, "type": "...", "confidence": 0.75, "tags": [...]}}
Only fill fields that have actual data. Omit fields with no data.
If no meaningful data can be extracted, return {{"cards": []}}.
Do NOT extract dashboard layout, metadata, or technical fields."""


async def _extract_tier3(responses: dict, source: dict, already_extracted: set[str]) -> list[dict]:
    """Tier 3: LLM-based extraction from remaining data not covered by T1/T2."""
    remaining = {
        k: v for k, v in responses.items()
        if k not in _SKIP_KEYS
        and k not in already_extracted
        and v is not None and v != {} and v != []
    }

    if not remaining:
        return []

    user_prompt = f"Runner completion data:\n{json.dumps(remaining, ensure_ascii=False, default=str)}"

    try:
        client = get_client_by_provider("openai")
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": _build_llm_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw = response.choices[0].message.content
        logger.info(f"Tier 3: LLM raw response: {raw[:500]}")
        parsed = json.loads(raw)

        if isinstance(parsed, list):
            items = parsed
        else:
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

    # Tier 1: _ai_context.data_semantics (deterministic, no LLM)
    tier1, extracted_paths = _extract_tier1(responses, source)

    # Tier 2: journey_contributions (deterministic, no LLM)
    tier2 = _extract_tier2(responses, source)

    proposals = tier1 + tier2

    # Tier 3: LLM fallback (only for data not already extracted by T1/T2)
    tier3 = []
    if not proposals:
        tier3 = await _extract_tier3(responses, source, extracted_paths)
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
