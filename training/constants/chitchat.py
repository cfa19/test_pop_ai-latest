"""
Chitchat Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
chitchat training data (greetings, thanks, farewells, etc.).
"""

from typing import List

# ==============================================================================
# CHITCHAT CATEGORIES
# ==============================================================================

CHITCHAT_CATEGORIES = {
    "greetings": {
        "name": "Greetings",
        "description": "Hello messages, greetings, conversation starters",
        "examples": [
            "hi",
            "hello",
            "hey there",
            "good morning",
            "what's up",
            "how are you"
        ]
    },
    "thanks": {
        "name": "Thanks & Appreciation",
        "description": "Thank you messages, appreciation, gratitude",
        "examples": [
            "thanks",
            "thank you",
            "appreciate it",
            "thanks so much",
            "thank you for your help"
        ]
    },
    "farewells": {
        "name": "Farewells",
        "description": "Goodbye messages, see you later, conversation endings",
        "examples": [
            "bye",
            "goodbye",
            "see you",
            "talk to you later",
            "have a good day",
            "take care"
        ]
    },
    "acknowledgments": {
        "name": "Acknowledgments",
        "description": "OK, got it, understood, confirmation messages",
        "examples": [
            "ok",
            "got it",
            "understood",
            "makes sense",
            "I see",
            "alright"
        ]
    },
    "small_talk": {
        "name": "Small Talk",
        "description": "Casual conversation, how are you, what's up, small talk",
        "examples": [
            "how's it going",
            "how are you doing",
            "what's new",
            "how have you been",
            "hope you're doing well"
        ]
    }
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural chitchat messages. Always respond with valid JSON."""


MESSAGE_GENERATION_PROMPT_TEMPLATE = """Generate {batch_size} diverse, natural chitchat messages for {category_name}.

Category: {category_name}
Description: {category_description}

CRITICAL REQUIREMENTS:

1. **VERY SHORT messages**: Generate messages ranging from:
   - SINGLE WORD: 1 word (e.g., "hi", "thanks", "bye")
   - VERY SHORT: 2-3 words (e.g., "thank you", "see ya", "got it")
   - SHORT: 4-6 words (e.g., "thanks for your help today", "how are you doing")
   - MEDIUM: 7-10 words (e.g., "thank you so much for all your help I really appreciate it")

2. **Natural typing variations** - Include realistic casual variations:
   - Case variations: "Hi", "hi", "HI", "Hi!", "hi!"
   - Punctuation: "hello", "hello!", "hello.", "hello!!", "hey there!"
   - Informal spelling: "thx", "ty", "ur", "u", "r", "k", "ok"
   - Casual: "hey", "heya", "yo", "sup", "wassup", "hiya"
   - Formal: "Good morning", "Thank you very much", "I appreciate your help"
   - Fragments: "thanks!", "appreciate it", "got it", "cool"
   - Multiple words: "see you later", "talk soon", "catch you later"

3. **Question variety** - Mix statement and question forms:
   - Statements: "thanks", "bye", "got it", "ok"
   - Questions: "how are you?", "what's up?", "how's it going?"
   - Exclamations: "hi!", "thanks!", "awesome!", "perfect!"

4. **Real casual language** - Sound like actual users:
   - Super casual: "hey", "sup", "thx", "k", "cool"
   - Casual: "hi there", "thanks a lot", "see ya", "alright"
   - Neutral: "hello", "thank you", "goodbye", "okay"
   - Polite: "thank you so much", "I really appreciate it", "have a great day"

Example messages showing variety:
- Single word: "hi"
- Single word: "thanks"
- 2 words: "thank you"
- 2 words: "got it"
- 3 words: "hey there friend"
- 4 words: "thanks for your help"
- 6 words: "I really appreciate all your help"
- 8 words: "thank you so much for helping me out today"
- With variations: "Hi!", "hi", "HI", "hello!", "Hello", "hey", "Hey there"
- Informal: "thx", "ty", "k", "ok", "bye", "see ya"

Generate {batch_size} unique messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return CHITCHAT_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return CHITCHAT_CATEGORIES[category_key]["description"]


def get_category_examples(category_key: str) -> List[str]:
    """Get examples for category."""
    return CHITCHAT_CATEGORIES[category_key]["examples"]


def build_message_generation_prompt(category_key: str, batch_size: int) -> str:
    """
    Build a prompt for generating chitchat messages.

    Args:
        category_key: Key from CHITCHAT_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in CHITCHAT_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = CHITCHAT_CATEGORIES[category_key]
    examples_str = "\n".join(f"  - {ex}" for ex in category["examples"])

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_examples=examples_str
    )
