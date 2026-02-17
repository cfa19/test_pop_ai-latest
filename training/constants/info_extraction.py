"""
Information Extraction Schemas and Prompts

Defines extraction schemas for different subcategories and prompt templates
for extracting structured information from user messages.
"""

# =============================================================================
# Extraction Schemas
# =============================================================================

EXTRACTION_SCHEMAS = {
    # =========================================================================
    # Professional context
    # =========================================================================
    "dream_roles": {
        "task": "Extract all desired jobs or roles mentioned in the message, including target companies and industries.",
        "fields": ["title", "company", "industry", "timeframe", "appeal"],
        "type": "fact",
    },
    "compensation_expectations": {
        "task": "Extract salary range expectation. If the user doesn't mention a currency, assume it's EUR.",
        "fields": ["minimum", "target", "maximum", "currency", "timeframe"],
        "type": "fact",
    },
    "desired_work_environment": {
        "task": "Extract preferences about work environment: remote/hybrid/in-office, company size, company stage, culture priorities, deal-breakers.",
        "fields": ["workMode", "companySize", "companyStage", "culturePriorities", "dealBreakers"],
        "type": "fact",
    },
    "career_change_considerations": {
        "task": "Extract EXPLICIT career change intentions ONLY if the user directly says they want to change careers, industries, or roles. Wanting a specific role (e.g. 'I want to be a CEO') is a dream role, NOT a career change. Only extract if the user explicitly mentions changing, switching, transitioning, or leaving their current career.",
        "fields": ["changeType", "riskTolerance", "payCutWillingness", "obstacles", "supportNeeded"],
        "type": "fact",
    },
    "job_search_status": {
        "task": "Extract job search status ONLY if the user explicitly says they are searching, applying, interviewing, or have received offers. Expressing a career aspiration (e.g. 'I want to be a CEO') does NOT mean they are actively searching. Only extract if the user directly mentions job searching activities.",
        "fields": ["currentlySearching", "urgency", "applications", "interviews", "offers", "desiredStartDate"],
        "type": "fact",
    },
    "skills": {
        "task": "Extract all skills, competencies, and abilities mentioned.",
        "fields": ["name", "level", "validationDate", "validationSource"],
        "type": "fact",
    },
    "experiences": {
        "task": "Extract job experiences including roles, companies, responsibilities, and achievements.",
        "fields": ["role", "company", "description", "startDate", "endDate"],
        "type": "fact",
    },
    "certifications": {
        "task": "Extract certifications, licenses, degrees, and formal qualifications mentioned.",
        "fields": ["name", "issuer", "date", "expiryDate"],
        "type": "fact",
    },
    "current_position": {
        "task": "Extract current job title, employer, and responsibilities.",
        "fields": ["title", "company", "startDate", "department"],
        "type": "fact",
    },

    # =========================================================================
    # Learning context
    # =========================================================================
    "knowledge": {
        "task": "Extract knowledge areas, domains of expertise, and educational background.",
        "fields": ["name", "level", "lastPracticed"],
        "type": "fact",
    },
    "learning_velocity": {
        "task": "Estimate the learning velocity of the person based on the message as a number between 0 and 100.",
        "fields": ["velocity"],
        "type": "fact",
    },
    "preferred_format": {
        "task": "Infer the preferred learning format of the person as a string from the following list: 'video', 'text', 'interactive', 'mentoring', 'practice', 'workshop'.",
        "fields": ["format"],
        "type": "fact",
    },
    "skill_aspirations": {
        "task": "Extract skills the person wants to learn or develop, including their learning plan, timeline, and current progress.",
        "fields": ["targetSkill", "learningPlan", "timeline", "progress"],
        "type": "fact",
    },
    "education_aspirations": {
        "task": "Extract educational goals: desired degrees, target institutions, timeline, and funding plans.",
        "fields": ["desiredDegree", "institution", "timeline", "funding"],
        "type": "fact",
    },
    "certification_aspirations": {
        "task": "Extract certification goals: target certifications, study plan, and planned exam date.",
        "fields": ["targetCert", "studyPlan", "examDate"],
        "type": "fact",
    },
    "skill_gaps": {
        "task": "Extract missing skills that are blocking career goals, including what aspiration they block.",
        "fields": ["missingSkill", "impact", "blockingAspiration", "aspirationType"],
        "type": "fact",
    },
    "knowledge_gaps": {
        "task": "Extract missing knowledge areas that are blocking career goals, including what aspiration they block.",
        "fields": ["missingKnowledge", "blockingAspiration", "aspirationType"],
        "type": "fact",
    },

    # =========================================================================
    # Social context
    # =========================================================================
    "mentors": {
        "task": "Extract information about mentors, coaches, advisors, and guidance relationships.",
        "fields": ["name", "expertise", "lastInteraction"],
        "type": "fact",
    },
    "journey_peers": {
        "task": "Extract information about peers, colleagues at similar level, and peer connections.",
        "fields": ["connectionType", "sharedGoals"],
        "type": "fact",
    },
    "people_helped": {
        "task": "Extract the number of people mentored, coached, or helped.",
        "fields": ["number"],
        "type": "fact",
    },
    "testimonials": {
        "task": "Extract feedback, recognition, and what others have said about them.",
        "fields": ["from", "text", "date"],
        "type": "fact",
    },
    "networking_activities": {
        "task": "Extract networking activities: conferences attended, coffee chats, events, people met, and follow-up actions.",
        "fields": ["activityType", "date", "peopleMet", "followUp"],
        "type": "fact",
    },
    "networking_goals": {
        "task": "Extract networking goals: people they want to meet, events they want to attend, networking strategy.",
        "fields": ["targetConnections", "targetEvents", "strategy"],
        "type": "fact",
    },
    "networking_preferences": {
        "task": "Extract networking preferences: preferred formats (1-on-1, groups, conferences), energy impact, networking style.",
        "fields": ["preferredFormats", "energyImpact", "style"],
        "type": "fact",
    },

    # =========================================================================
    # Psychological context
    # =========================================================================
    "personality_profile": {
        "task": "Extract the Myers-Briggs Type Indicator (MBTI) or BigFive personality traits, behavioral tendencies, and temperament descriptors.",
        "fields": ["type", "results", "assessmentDate"],
        "type": "fact",
    },
    "strengths": {
        "task": "Extract mentioned strengths and positive attributes.",
        "fields": ["name", "description"],
        "type": "fact",
    },
    "weaknesses": {
        "task": "Extract mentioned weaknesses, challenges, or areas for improvement.",
        "fields": ["name", "description"],
        "type": "fact",
    },
    "motivations": {
        "task": "Extract what motivates or drives the person, including intrinsic and extrinsic motivators and demotivators.",
        "fields": ["intrinsic", "extrinsic", "demotivators"],
        "type": "fact",
    },
    "work_style": {
        "task": "Extract work style preferences: how they work, collaborate, make decisions, and communicate.",
        "fields": ["workStyle", "collaborationStyle", "decisionMaking", "communicationStyle"],
        "type": "fact",
    },
    "values": {
        "task": """Infer which of the following values are mentioned in the message and assign a value between 0 and 100 to each
        depending on the strength of the importance given by the user to it:
        work-life balance, social impact, financial security, personal growth, creativity, autonomy.""",
        "fields": ["workLifeBalance", "socialImpact", "financialSecurity", "personalGrowth", "creativity", "autonomy"],
        "type": "fact",
    },
    "confidence_levels": {
        "task": "Extract confidence level indicators: overall confidence, domain-specific confidence (technical, social, leadership), and what affects confidence.",
        "fields": ["overallConfidence", "domainConfidence", "confidenceFactors", "recentChanges"],
        "type": "fact",
    },
    "imposter_syndrome_and_doubt": {
        "task": "Extract indicators of imposter syndrome, self-doubt, comparison patterns, self-efficacy, and resilience.",
        "fields": ["imposterLevel", "triggers", "frequency", "comparisonPatterns", "selfEfficacy"],
        "type": "fact",
    },
    "self_talk_and_validation": {
        "task": "Extract inner critic patterns, self-compassion level, need for external validation, and reaction to criticism/praise.",
        "fields": ["innerCriticStrength", "selfCompassion", "externalValidationNeed", "reactionToCriticism"],
        "type": "fact",
    },
    "confidence_building_strategies": {
        "task": "Extract strategies for building confidence: what helps, what hurts, current efforts, and goals.",
        "fields": ["strategiesThatHelp", "strategiesThatHurt", "currentEfforts", "confidenceGoals"],
        "type": "fact",
    },
    "stress": {
        "task": "Extract stress factors, sources of pressure, or burnout indicators.",
        "fields": ["name", "intensity", "frequency", "copingStrategy"],
        "type": "fact",
    },
    "confidence": {
        "task": "Extract indicators of confidence level, self-doubt, or imposter syndrome.",
        "fields": ["value", "context"],
        "type": "fact",
    },
    "energy_patterns": {
        "task": "Extract energy patterns, burnout, motivation cycles, when feeling most productive or drained.",
        "fields": ["type", "value", "timeOfDay", "notes"],
        "type": "fact",
    },
    "celebration_moments": {
        "task": "Extract celebration moments, recent wins, accomplishments, proud moments, things going well.",
        "fields": ["date", "description", "impact", "relatedContext"],
        "type": "fact",
    },

    # =========================================================================
    # Personal context
    # =========================================================================
    "physical_health": {
        "task": "Extract physical health information: overall health, chronic conditions, energy levels, and physical limitations.",
        "fields": ["overallHealth", "chronicConditions", "energyLevels", "limitations"],
        "type": "fact",
    },
    "mental_health": {
        "task": "Extract mental health information: conditions, severity, treatment status, and impact on work.",
        "fields": ["conditions", "severity", "treatment", "impactOnWork"],
        "type": "fact",
    },
    "addictions_or_recovery": {
        "task": "Extract addiction/recovery information: type, status (active/recovery), clean time, recovery program, triggers, and career impact.",
        "fields": ["addictionType", "status", "cleanSince", "recoveryProgram", "triggers", "careerImpact"],
        "type": "fact",
    },
    "overall_wellbeing": {
        "task": "Extract overall wellbeing indicators: stress level, wellbeing score, general state.",
        "fields": ["stressLevel", "wellbeingScore", "generalState"],
        "type": "fact",
    },
    "life_goals": {
        "task": "Extract personal life goals, lifestyle aspirations, and non-career ambitions mentioned.",
        "fields": ["title", "description", "targetDate", "progress"],
        "type": "fact",
    },
    "impact_legacy": {
        "task": "Extract statements about desired impact, legacy, helping others, or making a difference.",
        "fields": ["impact", "legacy", "whoToHelp"],
        "type": "fact",
    },
    "skill_expertise": {
        "task": "Extract statements about desired skill expertise, mastery, or development.",
        "fields": ["skill", "expertise", "development"],
        "type": "fact",
    },
}

# Aliases: model outputs these names but the schema lives under a different key
_SCHEMA_ALIASES = {
    "salary_expectations": "compensation_expectations",
}
for _alias, _target in _SCHEMA_ALIASES.items():
    if _target in EXTRACTION_SCHEMAS and _alias not in EXTRACTION_SCHEMAS:
        EXTRACTION_SCHEMAS[_alias] = EXTRACTION_SCHEMAS[_target]

# =============================================================================
# Prompt Templates
# =============================================================================

EXTRACTION_SYSTEM_MESSAGE = (
    "You are a strict information extraction assistant. "
    "Extract ONLY facts that are EXPLICITLY and DIRECTLY stated in the user's message. "
    "DO NOT infer, assume, or deduce information that is not clearly written. "
    "If the user says 'I want to be a CEO', that is a dream role — it does NOT mean they are "
    "actively searching for jobs, considering a career change, or have any job search status. "
    "When in doubt, return null. Return valid JSON only."
)


def build_extraction_prompt(schema: dict, message: str) -> str:
    """
    Build extraction prompt for a given message and schema.

    Args:
        message: User message to extract information from
        schema: Extraction schema with 'prompt' and 'fields' keys

    Returns:
        Formatted extraction prompt
    """
    import json

    return f"""Task: {schema['task']}
Message: "{message}"

Return a JSON array of objects with these fields:
{json.dumps(schema['fields'], indent=2)}

STRICT RULES:
- Extract ONLY information that is EXPLICITLY written in the message.
- DO NOT infer, assume, or deduce anything beyond what is directly stated.
- If the message does not contain clear information for a field, set it to null.
- If no relevant information exists for this task at all, return an empty array: []

Return ONLY the JSON array, no additional text."""


def format_extracted_data(subcategory: str, items: list[dict]) -> str:
    """
    Format an array of extracted objects into a formatted string using the schema's format template.

    Args:
        subcategory: Subcategory name (e.g., "experiences", "skills", etc.)
        items: Array of objects with fields matching the schema's format placeholders

    Returns:
        Formatted string with each item formatted according to the schema format

    Raises:
        KeyError: If subcategory is not found in EXTRACTION_SCHEMAS
        KeyError: If schema does not have a "format" field

    Example:
        experiences = [
            {
                "role": "Software Engineer",
                "company": "Tech Corp",
                "startDate": "2020-01",
                "endDate": "2022-06",
                "description": "Developed web applications"
            }
        ]
        result = format_extracted_data("experiences", experiences)
        # Returns: "Work Experience: Software Engineer at Tech Corp from 2020-01 to 2022-06: Developed web applications"
    """
    # Get schema for subcategory
    schema = EXTRACTION_SCHEMAS.get(subcategory)
    if schema is None:
        raise KeyError(f"Subcategory '{subcategory}' not found in EXTRACTION_SCHEMAS")

    # Check if format template exists
    format_spec = schema.get("format")
    if format_spec is None:
        raise KeyError(f"Schema for '{subcategory}' does not have a 'format' field")

    # Get expected fields from schema
    schema_fields = schema.get("fields", [])
    if not schema_fields:
        raise KeyError(f"Schema for '{subcategory}' does not have a 'fields' field")

    import re
    placeholder_pattern = r'\{(\w+)\}'

    # Collect all possible fields from all formats (for format selection)
    all_format_fields = set()
    all_format_fields.update(re.findall(placeholder_pattern, format_spec))

    # Add missing fields to each item before formatting
    items_with_fields = []
    for item in items:
        # Check which fields from schema["fields"] are present and which are lacking
        present_fields = [field for field in schema_fields if field in item and item[field]]
        lacking_fields = [field for field in schema_fields if field not in item or not item[field]]
        
        # Create a copy of the item and add missing schema fields with empty string
        item_complete = dict(item)
        for field in schema_fields:
            if field not in item_complete:
                item_complete[field] = ""
        
        # Also add any format fields that aren't in schema fields (for format compatibility)
        for field in all_format_fields:
            if field not in item_complete:
                item_complete[field] = ""
        
        items_with_fields.append(item_complete)

    formatted_items = []

    for item in items_with_fields:
        # Get non-empty fields in item
        available_fields = {k: v for k, v in item.items() if v and str(v).strip()}
        
        # Extract placeholders from chosen format
        required_fields = set(re.findall(placeholder_pattern, format_spec))
        
        # Build kwargs with all required fields (now guaranteed to exist in item)
        kwargs = {field: item.get(field, "") for field in required_fields}
        
        # Format using the chosen template
        formatted = format_spec.format(**kwargs)
        formatted_items.append(formatted)

    return "\n\n".join(formatted_items).strip()
