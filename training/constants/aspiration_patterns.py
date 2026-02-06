"""
Aspirational Linguistic Pattern Sub-Categories

Defines the 6 linguistic/psychological patterns for expressing career aspirations.
These patterns classify HOW aspirations are expressed, complementing the WHAT
classification in aspirational.py (content categories).

Pattern Hierarchy:
- Content Categories (WHAT): dream_roles, salary, life_goals, values, impact, skills
- Linguistic Patterns (HOW): explicit, future, intentional, identity, regret, conditional
"""

from typing import Dict, List

# ==============================================================================
# LINGUISTIC PATTERN SUB-CATEGORIES
# ==============================================================================

ASPIRATION_PATTERNS = {
    "explicit_desire": {
        "name": "Explicit Desire",
        "description": "Direct, unambiguous statements of wanting or desiring something",
        "characteristics": [
            "Uses desire verbs: want, wish, desire, hope",
            "Present tense expression of aspiration",
            "Clear and direct intent",
            "No conditional or uncertainty",
        ],
        "pattern_templates": [
            "I want to [OBJ]",
            "I wish to [OBJ]",
            "I desire to [OBJ]",
            "I really want to [OBJ]",
            "I'd like to [OBJ]",
            "I hope to [OBJ]",
            "I'm hoping to [OBJ]",
            "I aspire to [OBJ]",
            "My goal is to [OBJ]",
            "My aim is to [OBJ]",
        ],
        "examples": [
            "I want to become a data scientist",
            "I wish to lead a team of engineers",
            "I hope to work at a FAANG company",
            "I'd like to double my current salary",
            "I aspire to make a real impact in healthcare",
        ],
    },
    "future_oriented": {
        "name": "Future-Oriented",
        "description": "Aspirations framed as future possibilities or expectations",
        "characteristics": [
            "Uses future tense or temporal markers",
            "Implies progression or timeline",
            "Less immediate urgency than explicit desire",
            "Often includes time references",
        ],
        "pattern_templates": [
            "Someday I'll [OBJ]",
            "One day I'll [OBJ]",
            "Eventually I'll [OBJ]",
            "In the future, I'll [OBJ]",
            "I'll [OBJ] when the time is right",
            "I'm going to [OBJ]",
            "I will [OBJ]",
            "Down the line, I'll [OBJ]",
            "In a few years, I'll [OBJ]",
            "By 2030, I'll [OBJ]",
        ],
        "examples": [
            "Someday I'll run my own company",
            "One day I'll be a principal engineer",
            "Eventually I'll transition into management",
            "In the future, I'll work remotely from abroad",
            "By 2025, I'll have doubled my income",
        ],
    },
    "intentional_change": {
        "name": "Intentional Change",
        "description": "Active planning and preparation for a career transition",
        "characteristics": [
            "Uses action verbs indicating planning or transition",
            "Implies active steps being taken",
            "Present progressive or near-future planning",
            "Shows agency and commitment",
        ],
        "pattern_templates": [
            "I'm planning to [OBJ]",
            "I'm working towards [OBJ]",
            "I'm preparing to [OBJ]",
            "I'm in the process of [OBJ]",
            "I'm transitioning to [OBJ]",
            "I'm aiming to [OBJ]",
            "I'm getting ready to [OBJ]",
            "I'm building towards [OBJ]",
            "I'm positioning myself to [OBJ]",
            "I'm taking steps to [OBJ]",
        ],
        "examples": [
            "I'm planning to switch careers into design",
            "I'm working towards becoming a tech lead",
            "I'm preparing to launch my own startup",
            "I'm transitioning into product management",
            "I'm taking steps to become a machine learning engineer",
        ],
    },
    "identity_projection": {
        "name": "Identity Projection",
        "description": "Seeing oneself in a future role or identity",
        "characteristics": [
            "Uses identity framing: 'see myself as', 'imagine myself'",
            "Self-concept and self-image focused",
            "Visualizes future self",
            "Less about action, more about being",
        ],
        "pattern_templates": [
            "I see myself as a [OBJ]",
            "I see myself [OBJ]",
            "I imagine myself [OBJ]",
            "I envision myself [OBJ]",
            "I picture myself [OBJ]",
            "I can see myself [OBJ]",
            "I view myself as a [OBJ]",
            "I consider myself a future [OBJ]",
            "I identify with [OBJ]",
            "In my mind, I'm already [OBJ]",
        ],
        "examples": [
            "I see myself as a leader in AI ethics",
            "I imagine myself running a successful agency",
            "I envision myself working on climate tech",
            "I picture myself as a CTO in 5 years",
            "I see myself making a difference in education",
        ],
    },
    "regret_based": {
        "name": "Regret-Based",
        "description": "Aspirations expressed through past regrets or missed opportunities",
        "characteristics": [
            "References past decisions or missed paths",
            "Uses regret language: wish, should have, if only",
            "Counterfactual thinking",
            "Often includes self-reflection on choices",
        ],
        "pattern_templates": [
            "I wish I had [OBJ]",
            "I should have [OBJ]",
            "If only I had [OBJ]",
            "I regret not [OBJ]",
            "Looking back, I wish I had [OBJ]",
            "I should've [OBJ] when I had the chance",
            "If I could go back, I'd [OBJ]",
            "I missed the opportunity to [OBJ]",
            "I didn't [OBJ] and now I regret it",
            "It's too bad I didn't [OBJ]",
        ],
        "examples": [
            "I wish I had pursued medicine instead",
            "I should have studied computer science in college",
            "If only I had started my own business earlier",
            "I regret not taking that job at Google",
            "Looking back, I wish I had focused on leadership skills",
        ],
    },
    "conditional_hypothetical": {
        "name": "Conditional/Hypothetical",
        "description": "Aspirations contingent on circumstances or presented as possibilities",
        "characteristics": [
            "Uses conditional language: if, might, could, maybe",
            "Expresses uncertainty or dependence on conditions",
            "Lower commitment level",
            "Acknowledges external factors or constraints",
        ],
        "pattern_templates": [
            "If things go well, I might [OBJ]",
            "If [condition], I'd [OBJ]",
            "I might [OBJ]",
            "I could [OBJ]",
            "Maybe I'll [OBJ]",
            "I'd consider [OBJ]",
            "If the opportunity arises, I'd [OBJ]",
            "Depending on [condition], I might [OBJ]",
            "I wouldn't mind [OBJ]",
            "If circumstances allow, I'll [OBJ]",
        ],
        "examples": [
            "If things go well, I might start a startup",
            "If I get the funding, I'd launch my own product",
            "I might transition into consulting",
            "Maybe I'll pursue an MBA",
            "If the opportunity arises, I'd move into executive leadership",
        ],
    },
}


# ==============================================================================
# PATTERN WEIGHTS AND PRIORITIES
# ==============================================================================

# Psychological commitment/confidence levels (1-5, 5 = highest commitment)
PATTERN_COMMITMENT_LEVELS = {
    "intentional_change": 5,  # Actively working on it
    "explicit_desire": 4,  # Clear intent but less action
    "identity_projection": 3,  # Self-concept formed but less concrete
    "future_oriented": 3,  # Confident but distant
    "conditional_hypothetical": 2,  # Uncertain, depends on conditions
    "regret_based": 1,  # Reflective, past-focused
}


# Recommended distribution for balanced training data
PATTERN_DISTRIBUTION = {
    "explicit_desire": 0.25,  # 25% - Most common
    "future_oriented": 0.20,  # 20%
    "intentional_change": 0.20,  # 20%
    "identity_projection": 0.15,  # 15%
    "conditional_hypothetical": 0.12,  # 12%
    "regret_based": 0.08,  # 8% - Least common
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_pattern_info(pattern_key: str) -> Dict:
    """
    Get pattern information by key.

    Args:
        pattern_key: Key from ASPIRATION_PATTERNS

    Returns:
        Pattern dictionary with name, description, characteristics, etc.

    Raises:
        ValueError: If pattern key is invalid
    """
    if pattern_key not in ASPIRATION_PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern_key}. Valid patterns: {list(ASPIRATION_PATTERNS.keys())}")

    return ASPIRATION_PATTERNS[pattern_key]


def get_all_pattern_keys() -> List[str]:
    """
    Get list of all pattern keys.

    Returns:
        List of pattern keys
    """
    return list(ASPIRATION_PATTERNS.keys())


def get_pattern_name(pattern_key: str) -> str:
    """
    Get pattern display name.

    Args:
        pattern_key: Key from ASPIRATION_PATTERNS

    Returns:
        Pattern display name
    """
    return get_pattern_info(pattern_key)["name"]


def get_pattern_templates(pattern_key: str) -> List[str]:
    """
    Get pattern templates for a specific pattern.

    Args:
        pattern_key: Key from ASPIRATION_PATTERNS

    Returns:
        List of pattern templates with [OBJ] placeholder
    """
    return get_pattern_info(pattern_key)["pattern_templates"]


def get_pattern_commitment_level(pattern_key: str) -> int:
    """
    Get psychological commitment level for a pattern (1-5).

    Args:
        pattern_key: Key from ASPIRATION_PATTERNS

    Returns:
        Commitment level (1-5, 5 = highest)
    """
    return PATTERN_COMMITMENT_LEVELS.get(pattern_key, 3)


def get_recommended_sample_count(total_samples: int, pattern_key: str) -> int:
    """
    Get recommended number of samples for a pattern based on distribution.

    Args:
        total_samples: Total number of samples to generate
        pattern_key: Key from ASPIRATION_PATTERNS

    Returns:
        Recommended sample count
    """
    distribution = PATTERN_DISTRIBUTION.get(pattern_key, 1.0 / len(ASPIRATION_PATTERNS))
    return int(total_samples * distribution)


# ==============================================================================
# PATTERN VALIDATION
# ==============================================================================


def validate_pattern_coverage(messages: List[str]) -> Dict[str, int]:
    """
    Validate that messages cover all pattern types.

    Args:
        messages: List of generated messages

    Returns:
        Dictionary mapping pattern_key -> count of messages
    """
    # Simple heuristic-based detection
    pattern_counts = {key: 0 for key in ASPIRATION_PATTERNS.keys()}

    for message in messages:
        message_lower = message.lower()

        # Explicit desire
        if any(word in message_lower for word in ["i want", "i wish", "i hope", "i aspire", "i desire", "my goal"]):
            pattern_counts["explicit_desire"] += 1

        # Future-oriented
        elif any(word in message_lower for word in ["someday", "one day", "eventually", "in the future", "i'll", "i will"]):
            pattern_counts["future_oriented"] += 1

        # Intentional change
        elif any(phrase in message_lower for phrase in ["i'm planning", "i'm working towards", "i'm preparing", "i'm transitioning", "i'm aiming"]):
            pattern_counts["intentional_change"] += 1

        # Identity projection
        elif any(phrase in message_lower for phrase in ["i see myself", "i imagine myself", "i envision", "i picture myself"]):
            pattern_counts["identity_projection"] += 1

        # Regret-based
        elif any(phrase in message_lower for phrase in ["i wish i had", "i should have", "if only", "i regret", "looking back"]):
            pattern_counts["regret_based"] += 1

        # Conditional/hypothetical
        elif any(word in message_lower for word in ["if ", "might", "could", "maybe", "perhaps", "depending on"]):
            pattern_counts["conditional_hypothetical"] += 1

    return pattern_counts


# ==============================================================================
# DOCUMENTATION
# ==============================================================================


def print_pattern_summary():
    """Print a formatted summary of all aspiration patterns."""
    print("=" * 80)
    print("ASPIRATION LINGUISTIC PATTERNS")
    print("=" * 80)
    print("\nThese patterns classify HOW aspirations are expressed:\n")

    for i, (key, pattern) in enumerate(ASPIRATION_PATTERNS.items(), 1):
        print(f"{i}. {pattern['name'].upper()} ({key})")
        print(f"   {pattern['description']}")
        print(f"   Commitment Level: {PATTERN_COMMITMENT_LEVELS[key]}/5")
        print(f"   Recommended %: {PATTERN_DISTRIBUTION[key] * 100:.0f}%")
        print("\n   Examples:")
        for example in pattern["examples"][:3]:
            print(f'   - "{example}"')
        print()


if __name__ == "__main__":
    # Print pattern summary when run directly
    print_pattern_summary()

    # Example usage
    print("=" * 80)
    print("PATTERN TEMPLATE EXAMPLES")
    print("=" * 80)
    print("\nExample templates for each pattern:\n")

    for key in get_all_pattern_keys():
        pattern_name = get_pattern_name(key)
        templates = get_pattern_templates(key)
        print(f"{pattern_name}:")
        for template in templates[:5]:
            print(f"  - {template}")
        print()
