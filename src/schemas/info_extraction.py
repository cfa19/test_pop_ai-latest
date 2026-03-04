"""
Information Extraction Schemas and Prompts

Defines extraction schemas for each entity in the canonical taxonomy
(entity_prompts.py).  One schema per entity that carries structured,
storable information.

Contexts / entities covered:
  professional  — work_history, professional_aspirations, professional_achievements,
                  workplace_challenges, job_search_status
  learning      — knowledge_and_credentials, languages, learning_agenda
  social        — mentorship, recommendations, network_and_networking
  psychological — mindset_and_values, working_style_preferences, emotional_state
  personal      — life_situation, health_and_wellbeing, personal_projects,
                  personal_priorities

No schemas for chitchat, meta, rag_query, or off_topic — those categories
carry no structured user-profile data to extract.
"""

# =============================================================================
# Extraction Schemas
# =============================================================================

EXTRACTION_SCHEMAS = {
    # ==========================================================================
    # PROFESSIONAL CONTEXT
    # ==========================================================================
    "work_history": {
        "task": (
            "Extract all current and past work roles mentioned. "
            "For each role capture: job title, company, whether it is the current role, "
            "start and end dates, department, compensation (salary / total comp), "
            "key responsibilities, and notable achievements or impact."
        ),
        "fields": [
            "role",
            "company",
            "isCurrent",
            "startDate",
            "endDate",
            "department",
            "compensation",
            "responsibilities",
            "achievements",
            "location",
        ],
        "type": "experience",
    },
    "professional_aspirations": {
        "task": (
            "Extract career aspirations and future plans: desired next roles, target companies "
            "and industries, target timeframe, skill gaps to address, salary expectations "
            "(target, minimum, currency), preferred work environment (remote/hybrid, company size, "
            "IC vs management), and any career change considerations (type, risk tolerance, "
            "willingness to take a pay cut)."
        ),
        "fields": [
            "dreamRole",
            "targetCompany",
            "targetIndustry",
            "targetTimeframe",
            "skillGapsToAddress",
            "targetSalary",
            "minimumSalary",
            "currency",
            "workEnvironment",
            "careerChangeType",
            "riskTolerance",
        ],
        "type": "aspiration",
    },
    "professional_achievements": {
        "task": (
            "Extract professional accomplishments beyond core job duties: "
            "awards and recognitions (title, awarding organization, date), "
            "volunteer or community roles (organization, role, description), "
            "publications (title, type: article/paper/blog/book, date), "
            "and regulatory or professional licenses (name, issuing body, expiry)."
        ),
        "fields": ["type", "title", "organization", "date", "description"],
        "type": "experience",
    },
    "workplace_challenges": {
        "task": (
            "Extract current workplace difficulties: type of challenge "
            "(e.g. micromanagement, toxic culture, conflict with manager, lack of growth, "
            "unfair treatment, organizational dysfunction), severity, how long the issue "
            "has existed, impact on performance and wellbeing, and any actions already taken."
        ),
        "fields": [
            "challengeType",
            "severity",
            "duration",
            "impactOnPerformance",
            "actionsTaken",
        ],
        "type": "trait",
    },
    "job_search_status": {
        "task": (
            "Extract current job search activity: search status (actively searching, "
            "casually browsing, not looking), urgency level, number of applications sent, "
            "active interview processes, offers received, and desired start date."
        ),
        "fields": [
            "searchStatus",
            "urgencyLevel",
            "applicationsSent",
            "interviewsInProgress",
            "offersReceived",
            "desiredStartDate",
        ],
        "type": "preference",
    },
    # ==========================================================================
    # LEARNING CONTEXT
    # ==========================================================================
    "knowledge_and_credentials": {
        "task": (
            "Extract what they currently know and have earned. For each item capture its type "
            "(skill, expertise_domain, course, book, degree, certification) and relevant details: "
            "name, proficiency level, years of experience, institution or issuer, "
            "field of study, date earned, and expiry date if applicable."
        ),
        "fields": [
            "type",
            "name",
            "level",
            "yearsExperience",
            "institution",
            "field",
            "date",
            "expiryDate",
        ],
        "type": "competence",
    },
    "languages": {
        "task": (
            "Extract EACH language separately as its own object. "
            "For each language extract: name, proficiency level "
            "(A1-C2 scale: learning / basic / intermediate / fluent / native), "
            "and any language certifications or test scores mentioned "
            "(e.g. TOEFL, IELTS, DELF, TCF). "
            "Use 'learning' when someone says they are currently learning a language. "
            "IMPORTANT: If multiple languages are mentioned, return an array "
            "with one object per language, each with its OWN correct proficiency. "
            "Do NOT group languages into a single object. "
            "Example: 'I speak French fluently and I'm learning Mandarin' → "
            '[{"language": "French", "proficiency": "fluent"}, '
            '{"language": "Mandarin", "proficiency": "learning"}]'
        ),
        "fields": ["language", "proficiency", "certification", "score"],
        "type": "competence",
    },
    "learning_agenda": {
        "task": (
            "Extract what they want to learn or develop next: skill gaps blocking career goals, "
            "knowledge gaps, target skills or certifications to acquire "
            "(with target date and study plan), "
            "desired degrees or programs, preferred learning formats "
            "(video, text, hands-on, cohort-based, mentoring, workshop), "
            "available time per week, and learning budget."
        ),
        "fields": [
            "gapOrGoal",
            "description",
            "targetDate",
            "preferredFormat",
            "hoursPerWeek",
            "budget",
        ],
        "type": "aspiration",
    },
    # ==========================================================================
    # SOCIAL CONTEXT
    # ==========================================================================
    "mentorship": {
        "task": (
            "Extract mentorship relationships in both directions. "
            "For each relationship record: direction (mentor or mentee), "
            "the other person's name, role and organization, meeting frequency, "
            "areas of guidance, and impact or progress."
        ),
        "fields": [
            "direction",
            "name",
            "role",
            "organization",
            "frequency",
            "guidanceAreas",
            "impact",
        ],
        "type": "connection",
    },
    "recommendations": {
        "task": (
            "Extract testimonials and professional references. "
            "For testimonials: author name, author role, recommendation text, platform "
            "(e.g. LinkedIn), and permission to share. "
            "For references: name, role, relationship, and availability."
        ),
        "fields": [
            "type",
            "authorName",
            "authorRole",
            "text",
            "platform",
            "relationship",
            "available",
        ],
        "type": "connection",
    },
    "network_and_networking": {
        "task": (
            "Extract professional network information: specific connections mentioned "
            "(name, role, organization, relationship strength), communities or groups "
            "(name, type: online/in-person, engagement level), networking events attended "
            "or planned, networking goals, and personal networking preferences "
            "(1-on-1 vs large events, energizing or draining)."
        ),
        "fields": [
            "type",
            "name",
            "role",
            "organization",
            "engagementLevel",
            "networkingGoal",
            "preference",
        ],
        "type": "connection",
    },
    # ==========================================================================
    # PSYCHOLOGICAL CONTEXT
    # ==========================================================================
    "mindset_and_values": {
        "task": (
            "Extract personality traits, motivations, demotivators, and mindset indicators - "
            "internal character attributes, not decisions or goals. For each item record the "
            "category (personality_trait, intrinsic_motivation, extrinsic_motivation, "
            "demotivator, or mindset_indicator), the value or label, its strength or importance "
            "(0-100 if quantifiable), and a brief description from the message. "
            "Do NOT extract personal rules, non-negotiables, work-life balance preferences, "
            "or personal life goals - those belong to personal_priorities."
        ),
        "fields": ["category", "value", "strength", "description"],
        "type": "emotion",
    },
    "working_style_preferences": {
        "task": (
            "Extract work and collaboration style preferences. For each preference record the "
            "dimension (work_style, collaboration_style, decision_making, communication_style, "
            "ideal_environment, energizer, or stressor) and the specific preference or description."
        ),
        "fields": ["dimension", "preference", "description"],
        "type": "preference",
    },
    "emotional_state": {
        "task": (
            "Extract current psychological wellbeing signals. For each signal record the "
            "dimension (confidence, imposter_syndrome, self_doubt, self_talk, stress, "
            "emotional_awareness, or coping_strategy), a numeric value if mentioned "
            "(e.g. 7/10), the context or trigger, and any coping strategy referenced."
        ),
        "fields": ["dimension", "value", "context", "copingStrategy"],
        "type": "emotion",
    },
    # ==========================================================================
    # PERSONAL CONTEXT
    # ==========================================================================
    "life_situation": {
        "task": (
            "Extract personal life context - both factual circumstances and how they constrain "
            "or enable career options. Factual: life stage, approximate age or age range, "
            "relationship status, partner's situation, children (number and ages), "
            "other dependents, childcare arrangements, housing type (own/rent/with family), "
            "city and region, who they live with, relocation openness. "
            "Career impact: constraints (financial stress, debt, childcare costs, health "
            "restrictions, location ties, recovery schedule), enablers (family support, "
            "financial security, partner support), risk tolerance, income dependency, "
            "severity, and expected timeframe of the circumstance."
        ),
        "fields": [
            "attribute",
            "value",
            "detail",
            "type",
            "severity",
            "timeframe",
            "impactOnCareer",
        ],
        "type": "trait",
    },
    "health_and_wellbeing": {
        "task": (
            "Extract diagnosed medical conditions, physical impediments, and addiction or "
            "recovery facts - strictly objective health information. Include: diagnosed physical "
            "conditions (chronic illness, injury, disability, surgery recovery), diagnosed mental "
            "health conditions being treated (ADHD, clinical depression, bipolar, anxiety disorder, "
            "autism spectrum, OCD, PTSD), and addiction or recovery (substance or behaviour, "
            "current status, time sober, program participation, known triggers, schedule impact). "
            "Do NOT extract general stress, burnout, low confidence, or emotional states - "
            "those belong to emotional_state."
        ),
        "fields": [
            "condition",
            "type",
            "status",
            "duration",
            "treatment",
            "impactOnWork",
        ],
        "type": "trait",
    },
    "personal_projects": {
        "task": (
            "Extract personal and side projects: project name, description, type "
            "(career_related, hobby, creative, volunteer), their role in the project, "
            "skills being used or developed, weekly time commitment, and motivation."
        ),
        "fields": [
            "name",
            "description",
            "type",
            "role",
            "skills",
            "hoursPerWeek",
            "motivation",
        ],
        "type": "experience",
    },
    "personal_priorities": {
        "task": (
            "Extract concrete personal commitments, hard rules, and personal life goals - "
            "things the person actively protects or works toward outside of work. Include: "
            "non-negotiables stated as firm decisions (e.g. won't work weekends, must be home "
            "for dinner, remote work only), personal life goals with a target outcome "
            "(e.g. lose weight, buy a house, travel, spend more time with family), life "
            "priorities ranked explicitly (e.g. family comes first), and schedule flexibility "
            "needs driven by personal life (school runs, medical appointments, recovery meetings). "
            "Do NOT extract personality traits, motivations, or how they approach challenges - "
            "those belong to mindset_and_values."
        ),
        "fields": ["category", "priority", "description", "timeframe"],
        "type": "preference",
    },
}

# =============================================================================
# Subcategory → Category mapping (single source of truth)
# =============================================================================

SUBCATEGORY_TO_CATEGORY: dict[str, str] = {
    "work_history": "professional",
    "professional_aspirations": "professional",
    "professional_achievements": "professional",
    "workplace_challenges": "professional",
    "job_search_status": "professional",
    "knowledge_and_credentials": "learning",
    "languages": "learning",
    "learning_agenda": "learning",
    "mentorship": "social",
    "recommendations": "social",
    "network_and_networking": "social",
    "mindset_and_values": "psychological",
    "working_style_preferences": "psychological",
    "emotional_state": "psychological",
    "life_situation": "personal",
    "health_and_wellbeing": "personal",
    "personal_projects": "personal",
    "personal_priorities": "personal",
}

# Valid "category.subcategory" labels for classification (derived from mapping)
VALID_EXTRACTION_LABELS: list[str] = [f"{cat}.{sub}" for sub, cat in SUBCATEGORY_TO_CATEGORY.items()]

# =============================================================================
# Prompt Templates
# =============================================================================

EXTRACTION_SYSTEM_MESSAGE = (
    "You are a precise information extraction assistant for a career coaching app. "
    "You extract personal information ABOUT THE USER from their message. "
    "Rules: "
    "1. Extract ONLY information that the user states about THEMSELVES. "
    "2. Do NOT extract information about other people, organizations, or activities "
    "unless it directly describes the user's own situation. "
    "For example, 'teaching kids to code' means the user teaches — it does NOT mean "
    "the user has children. "
    "3. If a field is not explicitly stated about the user, set it to null. "
    "4. NEVER infer or guess values. If unsure, use null. "
    "5. If MULTIPLE distinct items are mentioned (e.g. two jobs, two languages), "
    "return a SEPARATE object for each in the array. "
    "Return valid JSON only."
)


def build_json_schema(fields: list[str]) -> dict:
    """
    Build a JSON Schema object from extraction field names.

    Used for Groq/OpenAI structured output enforcement — the model is
    constrained to return ONLY the defined fields.

    Returns:
        JSON Schema dict for ``{"items": [{field: str|null, ...}]}``
    """
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {f: {"type": ["string", "null"]} for f in fields},
                    "required": fields,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_extraction_prompt(schema: dict, message: str) -> str:
    """
    Build extraction prompt for a given message and schema.

    Args:
        message: User message to extract information from
        schema: Extraction schema with 'task' and 'fields' keys

    Returns:
        Formatted extraction prompt
    """
    import json

    json_schema = build_json_schema(schema["fields"])

    return f"""
Task: {schema["task"]}
Message: "{message}"

Return a JSON object matching this exact schema:
{json.dumps(json_schema, indent=2)}

Rules:
- If multiple distinct items are mentioned, return ONE object per item.
- Each field should be a single string value, NOT an array.
- If a field has no relevant information, set it to null.
- Example: {{"items": [{{{", ".join(f'"{f}": "..."' for f in schema["fields"])}}}]}}

Return ONLY valid JSON, no additional text."""


def format_extracted_data(subcategory: str, items: list[dict]) -> str:
    """
    Format an array of extracted objects into a formatted string
    using the schema's format template.

    Args:
        subcategory: Subcategory name (e.g., "work_history", "languages", etc.)
        items: Array of objects with fields matching the schema's format placeholders

    Returns:
        Formatted string with each item formatted according to the schema format

    Raises:
        KeyError: If subcategory is not found in EXTRACTION_SCHEMAS
        KeyError: If schema does not have a "format" field
    """
    schema = EXTRACTION_SCHEMAS.get(subcategory)
    if schema is None:
        raise KeyError(f"Subcategory '{subcategory}' not found in EXTRACTION_SCHEMAS")

    format_spec = schema.get("format")
    if format_spec is None:
        raise KeyError(f"Schema for '{subcategory}' does not have a 'format' field")

    schema_fields = schema.get("fields", [])
    if not schema_fields:
        raise KeyError(f"Schema for '{subcategory}' does not have a 'fields' field")

    import re

    placeholder_pattern = r"\{(\w+)\}"

    all_format_fields = set()
    all_format_fields.update(re.findall(placeholder_pattern, format_spec))

    items_with_fields = []
    for item in items:
        item_complete = dict(item)
        for field in schema_fields:
            if field not in item_complete:
                item_complete[field] = ""
        for field in all_format_fields:
            if field not in item_complete:
                item_complete[field] = ""
        items_with_fields.append(item_complete)

    formatted_items = []
    for item in items_with_fields:
        required_fields = set(re.findall(placeholder_pattern, format_spec))
        kwargs = {field: item.get(field, "") for field in required_fields}
        formatted = format_spec.format(**kwargs)
        formatted_items.append(formatted)

    return "\n\n".join(formatted_items).strip()
