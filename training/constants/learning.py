"""
Learning Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
learning context training data.
"""

from typing import Dict, List


# ==============================================================================
# LEARNING CATEGORIES
# ==============================================================================

LEARNING_CATEGORIES = {
    "knowledge": {
        "name": "Knowledge",
        "description": "WHAT YOU KNOW—current knowledge areas, expertise domains, educational background, subjects mastered, what you've learned. Focus on accumulated knowledge and expertise, not learning speed.",
        "examples": [
            "I have a strong background in computer science fundamentals",
            "I'm well-versed in cloud architecture and distributed systems",
            "My knowledge of machine learning is mostly self-taught",
            "I studied economics in college but pivoted to tech",
            "As a nurse, I've been learning new medical procedures like wound care and IV insertion",
            "I know Python, SQL, and have basic knowledge of JavaScript"
        ]
    },
    "learning_velocity": {
        "name": "Learning Velocity",
        "description": "HOW FAST YOU LEARN—learning speed, capacity to pick up new skills quickly, how long it takes you. Focus on SPEED and PACE of learning, not what you're learning or how you prefer to learn.",
        "examples": [
            "I'm a fast learner and can pick up new technologies quickly",
            "I need time to deeply understand concepts before moving forward—I'm a deliberate learner",
            "I can become productive with a new framework in about 2 weeks",
            "I pick things up pretty quickly once I get hands-on experience",
            "It usually takes me a few months to really master something new",
            "I'm a quick study—I absorbed the new software system in just a few days"
        ]
    },
    "preferred_format": {
        "name": "Preferred Learning Format",
        "description": "HOW YOU LEARN BEST—preferred learning methods, formats, resources (videos, books, courses, mentorship, hands-on practice). Focus on learning METHODS and FORMATS, not speed or knowledge content.",
        "examples": [
            "I learn best through video tutorials and online courses",
            "I prefer reading technical documentation and books",
            "Hands-on projects are the most effective way for me to learn",
            "I need a mentor or coach to guide my learning journey",
            "I learn best by doing—trial and error works for me",
            "I prefer structured courses over self-directed learning"
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
1. Include the placeholder [OBJ] where the learning detail goes
2. Be natural and conversational (first-person statements)
3. Be specifically relevant to {category_name_lower}
4. Vary in formality and detail level
5. Use appropriate tense (present for current state, past for history)

Context examples for this category:
{category_examples}

Examples of good patterns for learning statements:
- "I have [OBJ]"
- "My knowledge of [OBJ] is strong"
- "I learn [OBJ]"
- "I'm [OBJ]"
- "I prefer [OBJ]"
- "My background includes [OBJ]"

Return ONLY the patterns, one per line, no numbering or bullets.
Each pattern MUST contain exactly one [OBJ] placeholder."""


OBJECT_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse {category_name_lower} statements that someone might share with a career coach.

Category: {category_name}
Description: {category_description}

Examples of what to generate:
{category_examples}

Requirements:
1. Generate realistic, specific learning-related statements
2. Cover diverse learning styles and preferences
3. Be authentic and relatable
4. Be concrete and specific (not generic)
5. One statement per line

Return ONLY the statements, no numbering or bullets."""


PATTERN_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural, conversational patterns for learning self-descriptions.

Your patterns should:
- Feel authentic and natural
- Use first-person voice ("I", "My", etc.)
- Include the [OBJ] placeholder exactly once
- Vary in structure and formality
- Be appropriate for career coaching context"""


OBJECT_GENERATION_SYSTEM_PROMPT = """You are a career coaching expert generating realistic learning profiles.

Your statements should:
- Be specific and concrete
- Cover diverse learning styles and backgrounds
- Sound authentic and relatable
- Avoid overly academic or formal language
- Be suitable for career coaching context"""


MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural learning context messages. Always respond with valid JSON."""

# Category-specific guidance to prevent confusion
CATEGORY_GUIDANCE = {
    "knowledge": """
CRITICAL: KNOWLEDGE is WHAT YOU KNOW—accumulated knowledge, expertise, subjects learned.
Focus: Content of your knowledge (what subjects/skills you've mastered)

✓ CORRECT: What you know/have learned (content)
  - "I have a strong background in computer science fundamentals"
  - "I'm well-versed in cloud architecture and distributed systems"
  - "As a nurse, I've been learning new medical procedures like wound care"
  - "I know Python, SQL, and have basic JavaScript knowledge"
  - "My knowledge of electrical codes is comprehensive from years of experience"

✗ WRONG: Learning speed (that's learning_velocity!)
  - "I'm a fast learner" ← NO! That's how fast you learn (learning_velocity)
  - "I pick things up quickly" ← NO! That's learning speed (learning_velocity)
  - "I can master new software in 2 weeks" ← NO! That's pace (learning_velocity)

✗ WRONG: Learning methods (that's preferred_format!)
  - "I learn best through hands-on practice" ← NO! That's how you learn (preferred_format)
  - "I prefer video tutorials" ← NO! That's learning method (preferred_format)

Rule: Knowledge = WHAT YOU KNOW. Velocity = HOW FAST. Format = HOW YOU LEARN BEST.""",

    "learning_velocity": """
CRITICAL: LEARNING VELOCITY is HOW FAST YOU LEARN—speed, pace, how quickly you pick things up.
Focus: SPEED and PACE of learning (not what you're learning or how)

✓ CORRECT: Learning speed/pace
  - "I'm a fast learner and can pick up new technologies quickly"
  - "I need time to deeply understand—I'm a deliberate learner"
  - "I can become productive with a new framework in about 2 weeks"
  - "I pick things up pretty quickly once I get hands-on"
  - "It takes me a few months to really master something new"
  - "I'm a quick study—absorbed the new system in just days"

✗ WRONG: What you've learned (that's knowledge!)
  - "I've been learning new medical procedures" ← NO! That's what you know (knowledge)
  - "I know Python and SQL" ← NO! That's your knowledge (knowledge)
  - "I have a background in nursing" ← NO! That's accumulated knowledge (knowledge)

✗ WRONG: How you prefer to learn (that's preferred_format!)
  - "I learn best through hands-on practice" ← NO! That's learning method (preferred_format)
  - "I prefer video tutorials" ← NO! That's format preference (preferred_format)

Rule: Velocity = HOW FAST. Knowledge = WHAT. Format = HOW (method).""",

    "preferred_format": """
CRITICAL: PREFERRED FORMAT is HOW YOU LEARN BEST—methods, formats, resources you prefer.
Focus: Learning METHODS and FORMATS (not speed or content)

✓ CORRECT: Learning methods/formats
  - "I learn best through video tutorials and online courses"
  - "I prefer reading technical documentation and books"
  - "Hands-on projects are the most effective way for me to learn"
  - "I need a mentor or coach to guide my learning"
  - "I learn best by doing—trial and error works for me"
  - "I prefer structured courses over self-directed learning"

✗ WRONG: Learning speed (that's learning_velocity!)
  - "I'm a fast learner" ← NO! That's speed (learning_velocity)
  - "I pick things up quickly" ← NO! That's pace (learning_velocity)

✗ WRONG: What you know (that's knowledge!)
  - "I know Python and SQL" ← NO! That's accumulated knowledge (knowledge)
  - "I've learned medical procedures" ← NO! That's what you know (knowledge)

Rule: Format = HOW YOU LEARN (method). Velocity = HOW FAST. Knowledge = WHAT.""",
}


MESSAGE_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural learning messages for career coaching specifically about {category_name}.

Category: {category_name}
Description: {category_description}
{category_guidance}

Each message should:
1. Be a complete, natural sentence expressing learning context
2. Be specifically relevant to {category_name_lower}
3. Vary in formality and detail level
4. Be conversational as if spoken by a real person to a career coach
5. Use appropriate tense (present for current state, past for background)
6. Be practical and career-focused
7. Vary WIDELY in length: mix very short (5-10 words), short (11-15 words), medium (16-25 words), and long (26-40 words) messages - CREATE SUBSTANTIAL variety
8. Reflect perspectives from DIVERSE professions: nurses learning new medical procedures, teachers mastering new curriculum, electricians learning new codes, chefs expanding their skills, social workers taking certifications, mechanics learning new vehicle systems, etc. - NOT just tech/software learning
9. Include REALISTIC TYPOS in some messages: "i learn", "prefere", "complted", "certfication", "knowlege", "thru"

Context examples for this category:
{category_examples}

Example messages showing WIDE variety in length and fields:
- Very short (6 words): "I learn best through hands-on practice"
- Short (12 words): "I have my nursing degree and recently completed ICU specialty training"
- Medium (24 words): "I completed my teaching degree ten years ago and have taken several workshops on classroom management, but I still struggle with differentiated instruction"
- Long (37 words): "I'm a fast learner when it comes to hands-on technical work like the kind I do as an electrician, and I can usually pick up new codes and techniques pretty quickly by watching someone do it once or twice"
- With typos: "i learn best thru hands on practice", "I complted my certfication last year", "i prefere reading manuals"
- "My background is in culinary arts"

Generate {batch_size} unique, complete messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return LEARNING_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return LEARNING_CATEGORIES[category_key]["description"]


def get_category_examples(category_key: str) -> List[str]:
    """Get examples for category."""
    return LEARNING_CATEGORIES[category_key]["examples"]


def build_pattern_generation_prompt(category_key: str, batch_size: int) -> str:
    """Build prompt for generating patterns."""
    category = LEARNING_CATEGORIES[category_key]
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
    category = LEARNING_CATEGORIES[category_key]
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
    Build a prompt for generating complete learning messages (no patterns/objects).

    Args:
        category_key: Key from LEARNING_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in LEARNING_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = LEARNING_CATEGORIES[category_key]
    examples_str = "\n".join(f"- {ex}" for ex in category["examples"])
    guidance = CATEGORY_GUIDANCE.get(category_key, "")
    category_guidance = f"Important: {guidance}" if guidance else ""

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_name_lower=category["name"].lower(),
        category_examples=examples_str,
        category_guidance=category_guidance
    )
