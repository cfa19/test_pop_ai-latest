"""
Emotional Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
emotional context training data.
"""

from typing import Dict, List


# ==============================================================================
# EMOTIONAL CATEGORIES
# ==============================================================================

EMOTIONAL_CATEGORIES = {
    "confidence": {
        "name": "Confidence",
        "description": "Self-confidence levels, self-doubt, imposter syndrome, belief in abilities",
        "examples": [
            "I struggle with imposter syndrome and often doubt my abilities",
            "I'm confident in my technical skills but less so in leadership",
            "I feel very confident when working on backend systems",
            "I sometimes worry I'm not good enough for senior roles"
        ]
    },
    "energy_patterns": {
        "name": "Energy Patterns",
        "description": "Energy levels, burnout, motivation cycles, when feeling most productive or drained",
        "examples": [
            "I'm most energized in the morning and struggle in the afternoon",
            "I've been feeling burned out from the constant deadlines",
            "I get energized when I'm solving complex problems",
            "long meetings drain my energy significantly"
        ]
    },
    "stress_triggers": {
        "name": "Stress Triggers",
        "description": "What causes stress or anxiety, workplace stressors, challenging situations",
        "examples": [
            "I get stressed when expectations aren't clear",
            "tight deadlines and ambiguity really trigger my anxiety",
            "I struggle with stress when I have to present to large groups",
            "conflict with teammates is a major source of stress for me"
        ]
    },
    "celebration_moments": {
        "name": "Celebration Moments",
        "description": "Recent wins, accomplishments, proud moments, things going well",
        "examples": [
            "I recently shipped a feature that got great user feedback",
            "I'm proud that I finally got promoted to senior engineer",
            "my team successfully launched a product under tight deadlines",
            "I received recognition for mentoring junior developers"
        ]
    }
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

PATTERN_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse message patterns for expressing {category_name_lower}.

Category: {category_name}
Description: {category_description}

Each pattern should:
1. Include the placeholder [OBJ] where the emotional detail goes
2. Be natural and conversational (first-person statements)
3. Be specifically relevant to {category_name_lower}
4. Vary in formality and emotional intensity
5. Use appropriate tense (present for current state, past for events)

Context examples for this category:
{category_examples}

Examples of good patterns for emotional statements:
- "I feel [OBJ]"
- "I struggle with [OBJ]"
- "I'm [OBJ]"
- "I get stressed when [OBJ]"
- "I'm proud that [OBJ]"
- "I experience [OBJ]"

Return ONLY the patterns, one per line, no numbering or bullets.
Each pattern MUST contain exactly one [OBJ] placeholder."""


OBJECT_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse {category_name_lower} statements that someone might share with a career coach.

Category: {category_name}
Description: {category_description}

Examples of what to generate:
{category_examples}

Requirements:
1. Generate realistic, specific emotional statements
2. Cover diverse emotional experiences and intensities
3. Be authentic and vulnerable where appropriate
4. Be concrete and specific (not generic)
5. One statement per line

Return ONLY the statements, no numbering or bullets."""


PATTERN_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural, conversational patterns for emotional self-expression.

Your patterns should:
- Feel authentic and vulnerable
- Use first-person voice ("I", "My", etc.)
- Include the [OBJ] placeholder exactly once
- Vary in emotional intensity
- Be appropriate for career coaching context"""


OBJECT_GENERATION_SYSTEM_PROMPT = """You are a career coaching expert generating realistic emotional experiences.

Your statements should:
- Be specific and concrete
- Cover diverse emotional states and experiences
- Sound authentic and vulnerable
- Avoid overly clinical or therapeutic language
- Be suitable for career coaching context"""


MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural emotional context messages. Always respond with valid JSON."""


MESSAGE_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural emotional messages for career coaching specifically about {category_name}.

Category: {category_name}
Description: {category_description}

Each message should:
1. Be a complete, natural sentence expressing emotional context
2. Be specifically relevant to {category_name_lower}
3. Vary in formality and emotional intensity
4. Be conversational as if spoken by a real person to a career coach
5. Use appropriate tense (present for current state, past for events)
6. Be authentic and appropriately vulnerable
7. Focus on career-related emotions and experiences
8. Vary WIDELY in length: mix very short (5-10 words), short (11-15 words), medium (16-25 words), and long (26-40 words) messages - CREATE SUBSTANTIAL variety
9. Reflect perspectives from DIVERSE professions: nurses dealing with patient care stress, teachers managing classroom burnout, construction workers proud of completed projects, retail workers handling difficult customers, social workers managing emotional labor, artists celebrating breakthroughs, etc. - NOT just office/tech emotional experiences
10. Include REALISTIC TYPOS in some messages: "i feel", "furstrated", "streesed", "cofident", "somtimes", "dificult"

Context examples for this category:
{category_examples}

Example messages showing WIDE variety in length and fields:
- Very short (5 words): "I struggle with imposter syndrome"
- Short (13 words): "I feel confident in my clinical skills but nervous about taking on leadership"
- Medium (24 words): "I get really stressed when parents confront me about their child's grades or behavior, and I often replay those conversations in my head all evening"
- Long (38 words): "I've been feeling burned out from the emotional weight of my social work caseload because seeing so much trauma and hardship every day takes a real toll, and I'm not sure how much longer I can sustain this without better organizational support"
- With typos: "i feel furstrated with my job", "i'm somtimes streesed about work", "I feel cofident in my skills now"
- "Dealing with angry customers drains my energy"

Generate {batch_size} unique, complete messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return EMOTIONAL_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return EMOTIONAL_CATEGORIES[category_key]["description"]


def get_category_examples(category_key: str) -> List[str]:
    """Get examples for category."""
    return EMOTIONAL_CATEGORIES[category_key]["examples"]


def build_pattern_generation_prompt(category_key: str, batch_size: int) -> str:
    """Build prompt for generating patterns."""
    category = EMOTIONAL_CATEGORIES[category_key]
    examples_text = "\n".join(f"  - {ex}" for ex in category["examples"])

    return PATTERN_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_name_lower=category["name"].lower(),
        category_description=category["description"],
        category_examples=examples_text
    )


def build_object_generation_prompt(category_key: str, batch_size: int) -> str:
    """Build prompt for generating objects."""
    category = EMOTIONAL_CATEGORIES[category_key]
    examples_text = "\n".join(f"  - {ex}" for ex in category["examples"])

    return OBJECT_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_name_lower=category["name"].lower(),
        category_description=category["description"],
        category_examples=examples_text
    )


def build_message_generation_prompt(category_key: str, batch_size: int) -> str:
    """
    Build a prompt for generating complete emotional messages (no patterns/objects).

    Args:
        category_key: Key from EMOTIONAL_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in EMOTIONAL_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = EMOTIONAL_CATEGORIES[category_key]
    examples_str = "\n".join(f"- {ex}" for ex in category["examples"])

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_name_lower=category["name"].lower(),
        category_examples=examples_str
    )
