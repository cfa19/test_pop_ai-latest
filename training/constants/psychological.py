"""
Psychological Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
psychological context training data.
"""

from typing import Dict, List


# ==============================================================================
# PSYCHOLOGICAL CATEGORIES
# ==============================================================================

PSYCHOLOGICAL_CATEGORIES = {
    "personality_profile": {
        "name": "Personality Profile",
        "description": "Personality traits, behavioral tendencies, communication styles, temperament",
        "examples": [
            "I'm an introvert who prefers deep one-on-one conversations",
            "I'm detail-oriented and methodical in my approach",
            "I thrive in collaborative environments with lots of interaction",
            "I'm naturally curious and love exploring new ideas"
        ]
    },
    "values_hierarchy": {
        "name": "Values Hierarchy",
        "description": "Your internal compass—what matters most to you and in what order. PRIORITIES and PRINCIPLES that guide decision-making, especially when trade-offs appear. Answers: 'What comes first when two good things conflict?' Stable over time. Abstract priorities like integrity, autonomy, impact, stability.",
        "examples": [
            "Integrity and honesty are my top priorities—I won't compromise on ethics for success",
            "I value work-life balance above all else, even if it means slower career growth",
            "Making a positive impact is more important to me than money or status",
            "I prioritize continuous learning over job security—growth matters most",
            "Autonomy comes first for me—I need control over how I work",
            "Stability and predictability matter more than exciting challenges"
        ]
    },
    "motivations_core": {
        "name": "Core Motivations",
        "description": "Your engine—what drives you to act, choose, or persist. What makes you want to get up and do the work, what sustains effort when things get hard. Answers: 'Why do I do this?' More dynamic and situational than values. CONCRETE DRIVERS like solving problems, helping others, achieving results, gaining mastery.",
        "examples": [
            "I'm driven by solving complex technical challenges—that's what gets me excited each day",
            "Seeing tangible results motivates me—I need to see the impact of my work",
            "I'm energized by helping others succeed and watching them grow",
            "Recognition and praise drive me—I need acknowledgment for my contributions",
            "Mastering new skills is what keeps me engaged and motivated",
            "Competition pushes me—I perform best when I'm striving to be the best"
        ]
    },
    "working_styles": {
        "name": "Working Styles",
        "description": "Preferred work environment, pace, structure, collaboration preferences",
        "examples": [
            "I work best with clear structure and defined processes",
            "I prefer flexible schedules and remote work",
            "I thrive in fast-paced, dynamic environments",
            "I like having autonomy but regular check-ins with my team"
        ]
    }
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

PATTERN_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse message patterns for describing {category_name_lower}.

Category: {category_name}
Description: {category_description}

Each pattern should:
1. Include the placeholder [OBJ] where the psychological trait/value goes
2. Be natural and conversational (first-person statements)
3. Be specifically relevant to {category_name_lower}
4. Vary in formality and certainty level
5. Use present tense (describing current traits/preferences)

Context examples for this category:
{category_examples}

Examples of good patterns for psychological statements:
- "I am [OBJ]"
- "I tend to be [OBJ]"
- "My personality is [OBJ]"
- "I value [OBJ]"
- "I'm someone who [OBJ]"
- "I prefer [OBJ]"

Return ONLY the patterns, one per line, no numbering or bullets.
Each pattern MUST contain exactly one [OBJ] placeholder."""


OBJECT_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse {category_name_lower} statements that someone might share with a career coach.

Category: {category_name}
Description: {category_description}

Examples of what to generate:
{category_examples}

Requirements:
1. Generate realistic, specific psychological descriptions
2. Cover diverse personality types and preferences
3. Be authentic and relatable
4. Be concrete and specific (not generic)
5. One statement per line

Return ONLY the statements, no numbering or bullets."""


PATTERN_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural, conversational patterns for psychological self-descriptions.

Your patterns should:
- Feel authentic and natural
- Use first-person voice ("I", "My", etc.)
- Include the [OBJ] placeholder exactly once
- Vary in structure and formality
- Be appropriate for career coaching context"""


OBJECT_GENERATION_SYSTEM_PROMPT = """You are a career coaching expert generating realistic psychological profiles.

Your statements should:
- Be specific and concrete
- Cover diverse personality types and preferences
- Sound authentic and relatable
- Avoid jargon or overly clinical language
- Be suitable for career coaching context"""


MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural psychological context messages. Always respond with valid JSON."""

# Category-specific guidance to prevent confusion between values and motivations
CATEGORY_GUIDANCE = {
    "personality_profile": """
CRITICAL: PERSONALITY is WHO YOU ARE—inherent traits, temperament, behavioral tendencies.
Focus: Core characteristics that define you as a person (not preferences or environment needs)

✓ CORRECT: Inherent traits/characteristics (who you are)
  - "I'm an introvert who prefers deep one-on-one conversations"
  - "I'm detail-oriented and methodical in my approach"
  - "I'm naturally curious and love exploring new ideas"
  - "I'm a creative thinker who sees connections others miss"
  - "I tend to be analytical and data-driven"

✗ WRONG: Work preferences/environment needs (that's working_styles!)
  - "I struggle with rigid structures" ← NO! That's a work environment preference (working_styles)
  - "I work best with clear structure" ← NO! That's how you work best (working_styles)
  - "I prefer flexible schedules" ← NO! That's a work preference (working_styles)

Rule: Personality = WHO YOU ARE (traits). Working Styles = HOW YOU WORK BEST (preferences).""",

    "values_hierarchy": """
CRITICAL: VALUES are your INTERNAL COMPASS—abstract priorities and principles.
Focus: What matters MOST when trade-offs appear (decision-making criteria)

✓ CORRECT: Abstract priorities/principles (your compass)
  - "Integrity and honesty are my top priorities—I won't compromise ethics for success"
  - "I value work-life balance above all else, even if it means slower career growth"
  - "Making a positive impact is more important to me than money"
  - "Autonomy comes first—I need control over how I work"

✗ WRONG: Energy/drivers (that's motivations!)
  - "I'm driven by solving complex problems" ← NO! That's what energizes you (motivations)
  - "Recognition motivates me" ← NO! That's what drives action (motivations)
  - "I'm energized by helping others succeed" ← NO! That's your engine (motivations)

Rule: Values = WHAT MATTERS MOST (principles). Motivations = WHAT DRIVES ACTION (energy).""",

    "motivations_core": """
CRITICAL: MOTIVATIONS are your ENGINE—what drives you to act and sustains effort.
Focus: What gives you ENERGY, what makes you persist (the force behind action)

✓ CORRECT: Concrete drivers/energizers (your engine)
  - "I'm driven by solving complex technical challenges—that's what excites me each day"
  - "Seeing tangible results motivates me—I need to see impact"
  - "I'm energized by helping others succeed and watching them grow"
  - "Recognition and praise drive me—I need acknowledgment"
  - "Competition pushes me—I perform best when striving to be the best"

✗ WRONG: Abstract priorities (that's values!)
  - "Integrity is my top priority" ← NO! That's what matters most (values)
  - "I value autonomy above all else" ← NO! That's a principle (values)
  - "Work-life balance matters most to me" ← NO! That's a priority (values)

Rule: Values = WHAT MATTERS (direction). Motivations = WHAT DRIVES YOU (force).

Key distinction: Values set direction, Motivations supply force.""",

    "working_styles": """
CRITICAL: WORKING STYLES are HOW YOU WORK BEST—environment, structure, pace preferences.
Focus: What CONDITIONS and ENVIRONMENTS you need to perform well (not who you are)

✓ CORRECT: Work environment/structure preferences (how you work best)
  - "I work best with clear structure and defined processes"
  - "I prefer flexible schedules and remote work"
  - "I struggle with rigid structures that limit creativity"
  - "I thrive in fast-paced, dynamic environments"
  - "I need autonomy but regular check-ins with my team"
  - "I perform best with minimal supervision and high freedom"

✗ WRONG: Inherent traits (that's personality_profile!)
  - "I'm creative and artistic" ← NO! That's who you are (personality_profile)
  - "I'm detail-oriented" ← NO! That's a trait (personality_profile)
  - "I'm naturally curious" ← NO! That's a characteristic (personality_profile)

Rule: Personality = WHO YOU ARE. Working Styles = CONDITIONS YOU NEED to work well.""",
}


MESSAGE_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural psychological messages for career coaching specifically about {category_name}.

Category: {category_name}
Description: {category_description}
{category_guidance}

Each message should:
1. Be a complete, natural sentence expressing psychological context
2. Be specifically relevant to {category_name_lower}
3. Vary in formality and detail level
4. Be conversational as if spoken by a real person to a career coach
5. Use present tense (describing current traits/preferences)
6. Avoid overly clinical or jargon-heavy language
7. Vary WIDELY in length: mix very short (5-10 words), short (11-15 words), medium (16-25 words), and long (26-40 words) messages - CREATE SUBSTANTIAL variety
8. Reflect perspectives from DIVERSE professions: teachers, nurses, tradespeople, retail workers, artists, social workers, accountants, service workers, etc. - NOT just office/tech professionals
9. Include REALISTIC TYPOS in some messages: "i'm", "prefere", "teh", "peple", "somtimes", "wich"

Context examples for this category:
{category_examples}

Example messages showing WIDE variety in length and fields:
- Very short (7 words): "I'm introverted and prefer working alone"
- Short (13 words): "I value creativity and need freedom to express myself in my work"
- Medium (21 words): "I'm naturally detail-oriented which serves me well as an accountant, but I sometimes worry I miss the bigger picture"
- Long (38 words): "I tend to be very empathetic and emotionally invested in my work with clients, which makes me a good therapist but also means I struggle with boundaries and often take on too much of other people's pain"
- With typos: "i prefere working alone", "I value teh peple i work with", "i'm somtimes to detail oriented"
- "Helping others is my top priority"

Generate {batch_size} unique, complete messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return PSYCHOLOGICAL_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return PSYCHOLOGICAL_CATEGORIES[category_key]["description"]


def get_category_examples(category_key: str) -> List[str]:
    """Get examples for category."""
    return PSYCHOLOGICAL_CATEGORIES[category_key]["examples"]


def build_pattern_generation_prompt(category_key: str, batch_size: int) -> str:
    """Build prompt for generating patterns."""
    category = PSYCHOLOGICAL_CATEGORIES[category_key]
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
    category = PSYCHOLOGICAL_CATEGORIES[category_key]
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
    Build a prompt for generating complete psychological messages (no patterns/objects).

    Args:
        category_key: Key from PSYCHOLOGICAL_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in PSYCHOLOGICAL_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = PSYCHOLOGICAL_CATEGORIES[category_key]
    examples_str = "\n".join(f"- {ex}" for ex in category["examples"])

    # Get category-specific guidance if available
    guidance = CATEGORY_GUIDANCE.get(category_key, "")
    category_guidance = f"\n{guidance}" if guidance else ""

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_name_lower=category["name"].lower(),
        category_examples=examples_str,
        category_guidance=category_guidance
    )
