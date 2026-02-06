"""
Aspirational Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
aspirational training data.
"""

from typing import Dict, List

# ==============================================================================
# ASPIRATION CATEGORIES
# ==============================================================================

ASPIRATION_CATEGORIES = {
    "dream_roles": {
        "name": "Dream Roles",
        "description": "Career positions, titles, and professional roles",
        "examples": ["become a CTO", "become a data scientist", "lead a team of engineers", "work as a principal engineer"],
    },
    "salary_expectations": {
        "name": "Salary Expectations",
        "description": "Compensation, financial goals, and monetary aspirations",
        "examples": [
            "earn over $200k annually",
            "double my current salary",
            "get equity in a successful startup",
            "reach a $500k total compensation",
        ],
    },
    "life_goals": {
        "name": "Life Goals Beyond Career",
        "description": "Personal life, family, travel, hobbies, and lifestyle aspirations",
        "examples": [
            "have more time for my family",
            "travel the world while working remotely",
            "achieve financial independence",
            "pursue my hobbies alongside work",
        ],
    },
    "values": {
        "name": "Values",
        "description": "Work principles, work-life balance, ethics, and what matters most",
        "examples": [
            "achieve work-life balance",
            "work with integrity and honesty",
            "find meaning in my work",
            "maintain my mental health while advancing",
        ],
    },
    "impact_legacy": {
        "name": "Impact & Legacy",
        "description": "Making a difference, helping others, leaving a mark on the world",
        "examples": [
            "make a positive impact on society",
            "build products that help millions",
            "mentor the next generation of engineers",
            "solve important problems with technology",
        ],
    },
    "skill_expertise": {
        "name": "Skill & Expertise",
        "description": "Mastering technologies, becoming an expert, developing skills",
        "examples": [
            "master machine learning",
            "become an expert in distributed systems",
            "learn cloud architecture deeply",
            "become proficient in multiple programming languages",
        ],
    },
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

PATTERN_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse message patterns for expressing career aspirations specifically about {category_name}.

Category: {category_name}
Description: {category_description}

Each pattern should:
1. Include the placeholder [OBJ] where the aspiration goes
2. Be natural and conversational
3. Be specifically relevant to {category_name_lower}
4. Vary in formality, time frame, and certainty level

Context examples for this category:
{category_examples}

Examples of good patterns:
- "I aspire to [OBJ]"
- "My dream is to [OBJ]"
- "I want to [OBJ] within the next few years"
- "I'm working towards [OBJ]"
- "I'd love to [OBJ] someday"
- "My ultimate goal is to [OBJ]"
- "I hope to [OBJ] eventually"

Generate {batch_size} unique patterns as a JSON array. Return ONLY valid JSON with no additional text:
{{"patterns": ["pattern1", "pattern2", ...]}}"""


ASPIRATION_GENERATION_PROMPT_TEMPLATE = """Generate {num_aspirations} diverse career aspirations specifically for: {category_name}

Category: {category_name}
Description: {category_description}

Each aspiration should:
1. Be a verb phrase (infinitive form) that fits after "I want to..." or "My goal is to..."
2. Be specific and actionable
3. Be directly relevant to {category_name_lower}
4. Cover different aspects and scopes within this category

Examples for this category:
{category_examples}

Generate {num_aspirations} unique aspirations as a JSON array. Return ONLY valid JSON with no additional text:
{{"aspirations": ["aspiration1", "aspiration2", ...]}}"""


# ==============================================================================
# SYSTEM PROMPTS
# ==============================================================================

PATTERN_GENERATION_SYSTEM_PROMPT = "You are an expert at generating diverse natural language patterns. Always respond with valid JSON."

ASPIRATION_GENERATION_SYSTEM_PROMPT = "You are an expert at understanding career aspirations. Always respond with valid JSON."


# ==============================================================================
# PROMPT BUILDER FUNCTIONS
# ==============================================================================


def build_pattern_generation_prompt(category_key: str, batch_size: int) -> str:
    """
    Build a pattern generation prompt for a specific category.

    Args:
        category_key: Key from ASPIRATION_CATEGORIES
        batch_size: Number of patterns to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in ASPIRATION_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = ASPIRATION_CATEGORIES[category_key]

    # Format category examples
    examples_str = "\n".join(f"- {ex}" for ex in category["examples"])

    # Build the prompt
    prompt = PATTERN_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_name_lower=category["name"].lower(),
        category_examples=examples_str,
    )

    return prompt


def build_aspiration_generation_prompt(category_key: str, num_aspirations: int) -> str:
    """
    Build an aspiration generation prompt for a specific category.

    Args:
        category_key: Key from ASPIRATION_CATEGORIES
        num_aspirations: Number of aspirations to generate

    Returns:
        Formatted prompt string
    """
    if category_key not in ASPIRATION_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = ASPIRATION_CATEGORIES[category_key]

    # Format category examples
    examples_str = "\n".join(f"- {ex}" for ex in category["examples"])

    # Build the prompt
    prompt = ASPIRATION_GENERATION_PROMPT_TEMPLATE.format(
        num_aspirations=num_aspirations,
        category_name=category["name"],
        category_description=category["description"],
        category_name_lower=category["name"].lower(),
        category_examples=examples_str,
    )

    return prompt


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_category_info(category_key: str) -> Dict:
    """
    Get category information by key.

    Args:
        category_key: Key from ASPIRATION_CATEGORIES

    Returns:
        Category dictionary with name, description, examples

    Raises:
        ValueError: If category key is invalid
    """
    if category_key not in ASPIRATION_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}. Valid categories: {list(ASPIRATION_CATEGORIES.keys())}")

    return ASPIRATION_CATEGORIES[category_key]


def get_all_category_keys() -> List[str]:
    """
    Get list of all category keys.

    Returns:
        List of category keys
    """
    return list(ASPIRATION_CATEGORIES.keys())


def get_category_name(category_key: str) -> str:
    """
    Get category display name.

    Args:
        category_key: Key from ASPIRATION_CATEGORIES

    Returns:
        Category display name
    """
    return get_category_info(category_key)["name"]
