"""
Category Definitions and Constants

Contains intent category definitions, examples, and other data constants
used for synthetic data generation.
"""

# ==============================================================================
# STORE A CONTEXTS
# ==============================================================================

STORE_A_CONTEXTS = ["professional", "psychological", "learning", "social", "emotional", "aspirational"]

# ==============================================================================
# INTENT CATEGORIES
# ==============================================================================

INTENT_CATEGORIES = {
    "rag_query": {
        "description": "Factual questions seeking information or knowledge",
        "examples": [
            "What is machine learning?",
            "How do I write a resume?",
            "What are the best practices for Python?",
            "Explain the difference between AI and ML",
            "What courses do you offer?",
        ],
    },
    "professional": {
        "description": "Professional skills, experience, technical abilities, work history",
        "examples": [
            "I have 5 years of Python development experience",
            "I'm certified in AWS solutions architecture",
            "I've led teams of 10+ engineers",
            "My expertise is in data engineering and ETL pipelines",
            "I specialize in frontend development with React",
        ],
    },
    "psychological": {
        "description": "Personality traits, values, motivations, work style preferences",
        "examples": [
            "I value work-life balance above all else",
            "I'm motivated by solving complex problems",
            "I prefer working independently rather than in teams",
            "I'm a perfectionist and detail-oriented person",
            "I thrive under pressure and tight deadlines",
        ],
    },
    "learning": {
        "description": "Learning preferences, educational background, learning styles",
        "examples": [
            "I learn best through hands-on projects",
            "I prefer video tutorials over reading documentation",
            "I have a Computer Science degree from MIT",
            "I'm a visual learner and need diagrams to understand concepts",
            "I like to learn by teaching others",
        ],
    },
    "social": {
        "description": "Network, mentors, community, relationships, social connections",
        "examples": [
            "My mentor helped me navigate my career transition",
            "I'm part of the local Python developer community",
            "I have a strong network in the fintech industry",
            "I regularly attend tech meetups and conferences",
            "My former manager is now a CTO at a startup",
        ],
    },
    "emotional": {
        "description": "Emotional wellbeing, confidence, stress, anxiety, burnout, feelings",
        "examples": [
            "I'm feeling burned out from work",
            "I lack confidence in my technical abilities",
            "I'm anxious about changing careers",
            "I feel overwhelmed by the pace of technology changes",
            "I'm proud of my recent project success",
        ],
    },
    "aspirational": {
        "description": "Career goals, dreams, ambitions, future plans",
        "examples": [
            "I want to become a CTO in 5 years",
            "My goal is to work at a FAANG company",
            "I dream of starting my own tech startup",
            "I aspire to be a thought leader in AI",
            "I want to transition from frontend to full-stack development",
        ],
    },
    "chitchat": {
        "description": "Small talk and social interactions - greetings, pleasantries, conversational responses, acknowledgments",
        "examples": [
            "Hi there!",
            "Hello, how are you?",
            "Hey, what's up?",
            "Thanks a lot!",
            "Nice to meet you",
            "How's it going?",
            "I appreciate your help",
            "That's great!",
            "Sounds good",
            "I see, interesting",
            "Got it, thanks",
            "Have a great day!",
        ],
    },
    "off_topic": {
        "description": (
            "Completely off-topic messages unrelated to career coaching - weather, "
            "sports, cooking, movies, travel, hobbies, pets, etc. "
            "NOTE: Used for testing semantic gate only, NOT for BERT classifier training."
        ),
        "examples": [
            "What's the weather like today?",
            "Did you see the game last night?",
            "I made pasta for dinner and it turned out great",
            "Have you watched the latest Marvel movie?",
            "I'm planning a trip to Japan next summer",
            "I love playing guitar in my free time",
            "My cat is so funny when she chases her tail",
            "The mountains look beautiful this time of year",
        ],
    },
}


# ==============================================================================
# CHITCHAT EXAMPLES
# ==============================================================================

CHITCHAT_EXAMPLES = ["hi", "hello", "hey there", "how are you", "what's up", "thanks", "ok", "sure", "got it", "nice to meet you"]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_category_info(category: str) -> dict:
    """
    Get description and examples for a specific category.

    Args:
        category: One of the intent categories

    Returns:
        Dictionary with 'description' and 'examples' keys

    Raises:
        KeyError: If category doesn't exist
    """
    if category not in INTENT_CATEGORIES:
        raise KeyError(f"Category '{category}' not found. Valid categories: {list(INTENT_CATEGORIES.keys())}")
    return INTENT_CATEGORIES[category]


def get_all_categories() -> list:
    """
    Get list of all intent category names.

    Returns:
        List of category names
    """
    return list(INTENT_CATEGORIES.keys())


def get_training_categories() -> list:
    """
    Get list of categories used for model training (excludes chitchat).

    Returns:
        List of 7 training category names
    """
    return ["rag_query", "professional", "psychological", "learning", "social", "emotional", "aspirational"]


def get_store_a_contexts() -> list:
    """
    Get list of Store A context categories (excludes rag_query).

    Returns:
        List of 6 Store A context names
    """
    return ["professional", "psychological", "learning", "social", "emotional", "aspirational"]
