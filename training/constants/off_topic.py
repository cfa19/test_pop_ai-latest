"""
Off-Topic Category Constants and Prompt Templates

Contains category definitions and prompt templates for generating
off-topic training data (messages unrelated to career coaching).
"""

from typing import List

# ==============================================================================
# OFF-TOPIC CATEGORIES
# ==============================================================================

OFF_TOPIC_CATEGORIES = {
    "random_topics": {
        "name": "Random Topics",
        "description": "Random subjects unrelated to career: food, sports, hobbies, weather, entertainment, politics",
        "examples": [
            "chocolate",
            "cycling",
            "I love pizza",
            "what's the weather like",
            "my favorite movie is inception",
            "do you like soccer"
        ]
    },
    "personal_life": {
        "name": "Personal Life (Non-Career)",
        "description": "Personal activities and events unrelated to professional life",
        "examples": [
            "I baked a cake today",
            "my dog is sick",
            "going to the beach this weekend",
            "had a great dinner with friends",
            "watching netflix tonight"
        ]
    },
    "general_knowledge": {
        "name": "General Knowledge Questions",
        "description": "Factual questions about geography, history, science, trivia unrelated to careers",
        "examples": [
            "what's the capital of France",
            "how tall is Mount Everest",
            "when did World War 2 end",
            "what's the speed of light",
            "who invented the telephone"
        ]
    },
    "nonsensical": {
        "name": "Nonsensical Messages",
        "description": "Random words, gibberish, unclear messages, fragments",
        "examples": [
            "banana",
            "asdfgh",
            "purple monkey dishwasher",
            "????",
            "idk man just stuff"
        ]
    },
    "current_events": {
        "name": "Current Events & News",
        "description": "News, politics, sports events, celebrity gossip unrelated to professional development",
        "examples": [
            "did you see the game last night",
            "what do you think about the election",
            "the new iPhone looks cool",
            "latest Marvel movie was amazing",
            "gas prices are so high"
        ]
    }
}


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

MESSAGE_GENERATION_SYSTEM_PROMPT = """You are an expert at generating natural off-topic messages. Always respond with valid JSON."""


MESSAGE_GENERATION_PROMPT_TEMPLATE = """\
Generate {batch_size} diverse, natural OFF-TOPIC messages that are \
unrelated to career coaching for {category_name}.

Category: {category_name}
Description: {category_description}

CRITICAL REQUIREMENTS:

1. **UNRELATED to career coaching**: Messages must have NOTHING to do with:
   - Career development, job searching, professional skills
   - Work life, workplace issues, career transitions
   - Professional goals, networking, resume building
   - ANY aspect of professional coaching or guidance

   Instead, focus on:
   - Personal hobbies, food, entertainment, sports
   - Random topics, general knowledge, trivia
   - Daily life activities unrelated to work
   - Nonsensical or unclear messages
   - Current events, politics, pop culture

2. **WIDE length variation**: Generate messages ranging from:
   - SINGLE WORD: 1 word (e.g., "chocolate", "pizza", "cycling")
   - VERY SHORT: 2-5 words (e.g., "I love sushi", "going to beach")
   - SHORT: 6-12 words (e.g., "I baked a chocolate cake this afternoon it was delicious")
   - MEDIUM: 13-20 words (e.g., "my dog has been sick for the past few days \
and I'm taking him to the vet tomorrow morning")
   - LONG: 21-35+ words (e.g., "I've been really into gardening lately and \
I just planted some tomatoes and peppers in my backyard but I'm worried \
they might not get enough sunlight because of the tree nearby")

3. **Natural typing variations** - Include realistic casual variations:
   - Spelling errors: "i bkaed a choco cake", "resturant", "reciept", "definately"
   - Missing words: "i baked cake", "went store", "watching movie"
   - Grammar mistakes: "I bake a cake yesterday", "my dog are sick", "we was going"
   - Casual abbreviations: "rn" (right now), "idk" (I don't know), "omg", "lol", "tbh"
   - No punctuation: "i love pizza so much", "what time is it"
   - All lowercase: "chocolate is my favorite", "going to the beach"
   - Typos: "teh", "hte", "adn", "wnat"

4. **Real off-topic language** - Sound like actual random messages:
   - Random words: "banana", "purple", "mountains", "coffee"
   - Incomplete thoughts: "thinking about...", "maybe pizza?", "idk probably"
   - Questions: "what's the weather?", "do you like cats?", "how tall is everest?"
   - Statements: "I love chocolate", "my dog is cute", "pizza is life"
   - Events: "went to the movies yesterday", "having dinner with mom"
   - Opinions: "I think pineapple belongs on pizza", "cats are better than dogs"

Example messages showing WIDE variety:
- Single word: "chocolate"
- Single word: "cycling"
- 2 words: "love pizza"
- 3 words: "my dog barked"
- 5 words: "I baked a cake today"
- 8 words: "went to the beach with my friends yesterday"
- 12 words: "I've been watching a lot of Netflix shows lately can't stop"
- 18 words: "my favorite hobby is gardening and I just planted some tomatoes in my backyard but they're not growing well"
- 28+ words: "I really enjoy baking and yesterday I tried making a chocolate \
cake from scratch for the first time and it turned out pretty good although \
the frosting was a bit too sweet"
- With variations: "i baked a cake", "i bkaed a choco cake", "I bake a delicious cake today", "baked a cake today"
- Typos: "I wnet to teh store", "my freind loves pizza", "definately going tommorrow"

Generate {batch_size} unique off-topic messages as a JSON array. Return ONLY valid JSON with no additional text:
{{"messages": ["message1", "message2", ...]}}"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category_name(category_key: str) -> str:
    """Get display name for category."""
    return OFF_TOPIC_CATEGORIES[category_key]["name"]


def get_category_description(category_key: str) -> str:
    """Get description for category."""
    return OFF_TOPIC_CATEGORIES[category_key]["description"]


def get_category_examples(category_key: str) -> List[str]:
    """Get examples for category."""
    return OFF_TOPIC_CATEGORIES[category_key]["examples"]


def build_message_generation_prompt(category_key: str, batch_size: int) -> str:
    """
    Build a prompt for generating off-topic messages.

    Args:
        category_key: Key from OFF_TOPIC_CATEGORIES
        batch_size: Number of messages to generate in this batch

    Returns:
        Formatted prompt string
    """
    if category_key not in OFF_TOPIC_CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    category = OFF_TOPIC_CATEGORIES[category_key]
    examples_str = "\n".join(f"  - {ex}" for ex in category["examples"])

    return MESSAGE_GENERATION_PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category["name"],
        category_description=category["description"],
        category_examples=examples_str
    )
