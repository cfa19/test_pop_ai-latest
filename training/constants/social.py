"""
Social Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
social context training data.
"""

from typing import Dict, List


# ==============================================================================
# SOCIAL CATEGORIES
# ==============================================================================

SOCIAL_CATEGORIES = {
    "mentors": {
        "name": "Mentors",
        "description": "Current or past mentors, coaches, advisors, people who guided career development. VARY THE LANGUAGE—avoid overusing 'mentor'. Use alternatives: coach, advisor, guide, teacher, professor, supervisor who guided you, senior colleague who helped you, etc.",
        "examples": [
            "I've had a great coach who helped me transition into tech",
            "my manager at my last job was an amazing guide and advisor",
            "I don't currently have a mentor but would love to find someone to guide me",
            "I've worked with several coaches and advisors throughout my career in different areas",
            "My professor in college was instrumental in helping me find my path",
            "I had a senior colleague who really took me under their wing and taught me the ropes",
            "My supervisor has been a great teacher and helped me develop my leadership skills",
            "I've been fortunate to have advisors from different industries who offer diverse perspectives"
        ]
    },
    "journey_peers": {
        "name": "Journey Peers",
        "description": "SAME-LEVEL peers on similar career paths—horizontal relationships, NOT mentors. People at your level: classmates, fellow students, co-workers at similar seniority, colleagues you collaborate with as equals, friends in similar roles, cohort members, fellow apprentices, etc. Focus on PEER-TO-PEER connections, not guidance relationships.",
        "examples": [
            "I'm part of a community of backend engineers who meet regularly and share experiences",
            "I have several peers who are also transitioning into product management—we support each other",
            "my network includes people from my bootcamp who are now at top tech companies",
            "I've built strong relationships with colleagues at my level across different companies",
            "I stay in touch with classmates from nursing school—we're all working at different hospitals now",
            "I have a group of fellow teachers I met at orientation who I lean on for support",
            "my cohort from the apprenticeship program still meets up to share what we're learning on the job",
            "I'm connected with other social workers in the area who are at a similar stage in their careers"
        ]
    },
    "people_helped": {
        "name": "People Helped",
        "description": "People mentored, coached, or helped in their careers, mentees, direct reports",
        "examples": [
            "I've mentored three junior engineers who are now mid-level",
            "I regularly help bootcamp graduates break into tech",
            "I've managed a team of 5 people and helped them grow their careers",
            "I mentor women in tech through a local organization"
        ]
    },
    "testimonials": {
        "name": "Testimonials",
        "description": "What others say about you, feedback received, recognition, reputation",
        "examples": [
            "people often tell me I'm a great communicator and teacher",
            "my teammates say I bring positive energy and help everyone collaborate better",
            "I received feedback that I'm excellent at breaking down complex problems",
            "my manager said I'm one of the most reliable people on the team"
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
1. Include the placeholder [OBJ] where the social detail goes
2. Be natural and conversational (first-person statements)
3. Be specifically relevant to {category_name_lower}
4. Vary in formality and detail level
5. Use appropriate tense (present for current, past for history)

Context examples for this category:
{category_examples}

Examples of good patterns for social statements:
- "I have [OBJ]"
- "My network includes [OBJ]"
- "I've [OBJ]"
- "People say I'm [OBJ]"
- "I'm part of [OBJ]"
- "I've helped [OBJ]"

Return ONLY the patterns, one per line, no numbering or bullets.
Each pattern MUST contain exactly one [OBJ] placeholder."""


OBJECT_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse {category_name_lower} statements that someone might share with a career coach.

Category: {category_name}
Description: {category_description}

Examples of what to generate:
{category_examples}

Requirements:
1. Generate realistic, specific social context statements
2. Cover diverse networking situations and relationships
3. Be authentic and relatable
4. Be concrete and specific (not generic)
5. One statement per line

Return ONLY the statements, no numbering or bullets."""


PATTERN_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural, conversational patterns for social context descriptions.

Your patterns should:
- Feel authentic and natural
- Use first-person voice ("I", "My", etc.)
- Include the [OBJ] placeholder exactly once
- Vary in structure and formality
- Be appropriate for career coaching context"""


OBJECT_GENERATION_SYSTEM_PROMPT = """You are a career coaching expert generating realistic social profiles.

Your statements should:
- Be specific and concrete
- Cover diverse networking situations
- Sound authentic and relatable
- Avoid overly formal or corporate language
- Be suitable for career coaching context"""


MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural social context messages. Always respond with valid JSON."""

# Category-specific guidance to prevent overusing certain words and category confusion
CATEGORY_GUIDANCE = {
    "mentors": """
CRITICAL: MENTORS are people who GUIDE, TEACH, or ADVISE you (vertical/guidance relationship).
Focus: People MORE SENIOR or EXPERIENCED who help you grow (not same-level peers)

✓ CORRECT format: Use VARIED terms for people who guided you
  - "I've had a great coach who helped me transition into tech"
  - "My professor was instrumental in helping me find my path"
  - "I had a senior colleague who took me under their wing"
  - "My supervisor has been a great teacher and guide"
  - "I've worked with several advisors throughout my career"
  - "My manager was an amazing guide when I was starting out"
  - "I don't currently have a mentor but would love to find someone to guide me"

Varied vocabulary: "coach", "advisor", "guide", "teacher", "professor", "supervisor who guided me", "senior colleague who helped me", "manager who taught me"

✗ WRONG format: Overusing "mentor/mentored"
  - "I've had several mentors and they mentored me in..." ← Too repetitive!
  - "My mentor mentored me through..." ← Use "My coach guided me" or "My advisor helped me"

Rule: Mentors = VERTICAL (guidance). Journey Peers = HORIZONTAL (same-level).""",

    "journey_peers": """
CRITICAL: JOURNEY PEERS are SAME-LEVEL connections (horizontal/peer-to-peer relationship).
Focus: People AT YOUR LEVEL, not mentors/guides who are more senior

✓ CORRECT format: Same-level peers and colleagues (horizontal relationships)
  - "I'm part of a community of nurses at my experience level who meet regularly"
  - "I have several peers who are also transitioning into management—we support each other"
  - "I stay in touch with classmates from nursing school—we're all working at different hospitals"
  - "I have a group of fellow teachers I met at orientation who I lean on for support"
  - "my cohort from the program still meets up to share what we're learning"
  - "I'm connected with other electricians who started around the same time I did"
  - "my network includes people from my bootcamp who are at similar stages in their careers"

Key phrases: "peers", "classmates", "colleagues at my level", "fellow [profession]", "cohort", "at a similar stage"

✗ WRONG format: Mentors or guidance relationships (that's mentors subcategory!)
  - "My mentor from the nursing field has helped me navigate..." ← NO! That's a mentor (guidance)
  - "I have a coach who guides me through..." ← NO! That's mentors (vertical relationship)
  - "My professor has been instrumental..." ← NO! That's a mentor (more senior)
  - "A senior colleague took me under their wing..." ← NO! That's mentors (guidance)

Rule: Journey Peers = HORIZONTAL (same-level, equals). Mentors = VERTICAL (guidance, more senior).""",
}


MESSAGE_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural social messages for career coaching specifically about {category_name}.

Category: {category_name}
Description: {category_description}
{category_guidance}

Each message should:
1. Be a complete, natural sentence expressing social context
2. Be specifically relevant to {category_name_lower}
3. Vary in formality and detail level
4. Be conversational as if spoken by a real person to a career coach
5. Use appropriate tense (present for current, past for history)
6. Be authentic and relatable
7. Vary WIDELY in length: mix very short (5-10 words), short (11-15 words), medium (16-25 words), and long (26-40 words) messages - CREATE SUBSTANTIAL variety
8. Reflect perspectives from DIVERSE professions: teachers connecting with educators, nurses supporting each other, tradespeople in unions, retail workers forming relationships, artists in creative communities, social workers in professional networks, etc. - NOT just tech/corporate networking
9. Include REALISTIC TYPOS in some messages: "i have", "collegues", "teh", "freind", "mentro", "thier"

Context examples for this category:
{category_examples}

Example messages showing WIDE variety in length and fields:
- Very short (8 words): "I mentor new nurses on my unit regularly"
- Short (14 words): "I'm part of a local teachers' group where we share lesson plans and resources"
- Medium (23 words): "My colleagues often tell me I'm good at explaining complex procedures to patients and families in a way that makes them feel comfortable"
- Long (36 words): "I've been mentoring apprentice electricians through our union's training program for about five years now, and it's incredibly rewarding to watch them develop their skills and confidence as they work toward becoming licensed journeymen"
- With typos: "i have a mentro who helps me", "my collegues say i'm good at training", "i don't have many freinds at work"
- "My manager says I train new hires well"

Generate {batch_size} unique, complete messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return SOCIAL_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return SOCIAL_CATEGORIES[category_key]["description"]


def get_category_examples(category_key: str) -> List[str]:
    """Get examples for category."""
    return SOCIAL_CATEGORIES[category_key]["examples"]


def build_pattern_generation_prompt(category_key: str, batch_size: int) -> str:
    """Build prompt for generating patterns."""
    category = SOCIAL_CATEGORIES[category_key]
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
    category = SOCIAL_CATEGORIES[category_key]
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
    Build a prompt for generating complete social messages (no patterns/objects).

    Args:
        category_key: Key from SOCIAL_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in SOCIAL_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = SOCIAL_CATEGORIES[category_key]
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
