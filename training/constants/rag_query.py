"""
RAG Query Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
RAG query training data based on knowledge base content.
"""

from typing import List

# ==============================================================================
# RAG QUERY CATEGORIES
# ==============================================================================

RAG_QUERY_CATEGORIES = {
    "company_overview": {
        "name": "Company Overview",
        "description": "Questions about Pop Skills/Activity Harmonia, mission, vision, market positioning",
        "topics": [
            "what is Pop Skills",
            "what is Activity Harmonia",
            "company mission and vision",
            "market positioning as Netflix of professional training",
            "number of employees and consultants",
            "certifications (Qualiopi, ISO)",
            "target for 2030"
        ]
    },
    "products": {
        "name": "Products & Services",
        "description": "Questions about PopCoach, PopSkills, SENSEI, and their features",
        "topics": [
            "what is PopCoach",
            "what is PopSkills",
            "what is SENSEI",
            "PopCoach pricing and trial",
            "PopCoach vs meditation apps",
            "PopCoach 12 runners",
            "PopSkills features",
            "difference between products"
        ]
    },
    "runners": {
        "name": "Runners System",
        "description": "Questions about runners, patterns, examples, and how they work",
        "topics": [
            "what are runners",
            "runner patterns (evaluation, accompaniment, validation, marketing, social, ritual)",
            "specific runners (Skills X-Ray, CV Optimizer, Interview Simulator, VAE)",
            "runner credits cost",
            "how runners work",
            "runner journey stages"
        ]
    },
    "programs": {
        "name": "Programs & Pricing",
        "description": "Questions about B2C and B2B programs, pricing, and packages",
        "topics": [
            "Five-Day Career Sprint",
            "Serene Career Change program",
            "Accelerated VAE program",
            "B2B programs (Onboarding Excellence, Talent Retention)",
            "program pricing",
            "CPF eligibility"
        ]
    },
    "credits_system": {
        "name": "Credits System",
        "description": "Questions about universal credits, how they work, and pricing",
        "topics": [
            "what are credits",
            "how credits work",
            "credit cost by pattern",
            "how to earn credits",
            "credit pricing"
        ]
    },
    "philosophy": {
        "name": "Philosophy & Ethics",
        "description": "Questions about company philosophy, AI ethics, and principles",
        "topics": [
            "augmented humanity principle",
            "AI ethics charter",
            "five guiding principles",
            "four dimensions of professional well-being",
            "ethics committee",
            "data privacy and GDPR"
        ]
    },
    "transformation_index": {
        "name": "Transformation Index",
        "description": "Questions about how transformation is measured and tracked",
        "topics": [
            "what is transformation index",
            "how is transformation measured",
            "TI formula and components",
            "six stages of transformation",
            "Lives Transformed Significantly metric"
        ]
    },
    "canonical_profile": {
        "name": "Canonical Profile & Contexts",
        "description": "Questions about the 6 contexts and user profile system",
        "topics": [
            "what is canonical profile",
            "6 contexts (professional, psychological, learning, social, emotional, aspirational)",
            "how profile data is used",
            "profile attributes"
        ]
    }
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural RAG query messages. Always respond with valid JSON."""


MESSAGE_GENERATION_PROMPT_TEMPLATE = """\
Generate {batch_size} diverse, natural questions about {category_name} \
that users would ask a career coaching chatbot.

Category: {category_name}
Description: {category_description}

Topics to cover:
{category_topics}

CRITICAL REQUIREMENTS:

1. **EXTREME length variation**: Generate questions ranging from:
   - VERY SHORT: 2-5 words (e.g., "what's popskills?", "popcoach pricing?")
   - SHORT: 6-10 words (e.g., "how much does popcoach cost per month")
   - MEDIUM: 11-18 words (e.g., "can you explain what the transformation \
index is and how it measures my progress")
   - LONG: 19-30+ words (e.g., "I'm interested in learning more about the \
different runner patterns you mentioned - could you explain what evaluation \
runners are versus accompaniment runners and give me some examples of each")

2. **Natural typing variations** - Include realistic user variations:
   - Case variations: "What is PopSkills?", "what is popskills", "WHAT IS POPSKILLS"
   - Punctuation: "what's popskills?", "what's popskills", "whats popskills"
   - Spacing: "pop skills", "popskills", "pop-skills"
   - Misspellings: "popskils", "pop coach", "popcoatch"
   - Abbreviations: "what's", "whats", "what is"
   - Informal: "tell me about popskills", "popskills info", "explain popskills"
   - Typos: "what's pposkills", "wat is popskills", "waht is pop coach", "hwo does it work", "priceing"

3. **Question variety** - Mix different question types:
   - Direct: "what is PopCoach"
   - Polite: "could you tell me about PopCoach"
   - Comparative: "what's the difference between PopCoach and PopSkills"
   - Detailed: "can you explain how the credits system works"
   - Brief: "popcoach features?"
   - Conversational: "I heard about PopCoach, what is it exactly"

4. **Real user language** - Make questions sound like actual users:
   - With context: "I'm considering signing up for PopCoach but want to know more about the pricing"
   - With uncertainty: "I think there are different products? what are they"
   - With intent: "looking for info on the five day career sprint program"
   - Fragment style: "popcoach trial limitations"
   - Complete sentences: "Can you explain what the canonical profile system is and how it works"

Example questions showing WIDE variety:
- Very short (2 words): "popskills pricing"
- Very short (3 words): "what's popcoach"
- Short (8 words): "how much does the career sprint program cost"
- Medium (15 words): "can you tell me what the difference is between PopCoach and the other products you offer"
- Long (28 words): "I'm really interested in understanding how the \
transformation index works and how you actually measure whether someone \
has been successfully transformed through using the platform and what \
criteria you use"
- With typos: "what's pposkills", "wat is pop coach", "hwo much does it cost", "can you expalin the priceing"

Generate {batch_size} unique questions as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["question1", "question2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return RAG_QUERY_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return RAG_QUERY_CATEGORIES[category_key]["description"]


def get_category_topics(category_key: str) -> List[str]:
    """Get topics for category."""
    return RAG_QUERY_CATEGORIES[category_key]["topics"]


def build_message_generation_prompt(category_key: str, batch_size: int) -> str:
    """
    Build a prompt for generating RAG query questions.

    Args:
        category_key: Key from RAG_QUERY_CATEGORIES
        batch_size: Number of questions to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in RAG_QUERY_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = RAG_QUERY_CATEGORIES[category_key]
    topics_str = "\n".join(f"  - {topic}" for topic in category["topics"])

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_topics=topics_str
    )
