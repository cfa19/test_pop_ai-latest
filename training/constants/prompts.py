"""
Prompt Templates for Synthetic Data Generation

Contains all text prompts used to generate training data via GPT-3.5-turbo.
Each prompt is a template string that can be formatted with specific parameters.
"""

# System message used for all API calls
SYSTEM_MESSAGE = (
    "You are a helpful assistant that generates realistic training data. "
    "Always respond with valid JSON. Escape all quotes inside strings with backslash. "
    "Your response must be a JSON array of strings."
)


# ==============================================================================
# RAG QUERY PROMPTS
# ==============================================================================

RAG_QUERY_GENERIC_PROMPT = """Generate {batch_size} diverse factual questions that someone might ask a career coaching chatbot.

Questions should be information-seeking queries about:
- Career advice and guidance
- Technical skills and technologies
- Resume and interview tips
- Industry trends and insights
- Learning resources and courses

Requirements for LENGTH VARIANCE (distribute evenly):
- SHORT (30%): 1 sentence, 5-15 words - Direct, simple questions
  Example: "What is Python?", "How do I start coding?"
- MEDIUM (50%): 1-2 sentences, 15-40 words - Questions with context
  Example: "I'm interested in data science. What programming languages should I learn first?"
- LONG (20%): 2-4 sentences, 40-80 words - Detailed questions with background
  Example: "I've been working as a software engineer for 3 years but I want to transition into machine learning. What skills should I focus on learning, and are there specific courses or certifications you'd recommend for someone with my background?"

Additional requirements:
- Vary question types (what, how, why, when, can you, should I, etc.)
- Include both beginner and advanced questions
- Make them specific and realistic
- Escape quotes properly (use \\" for quotes inside strings)

Return ONLY a valid JSON array of strings, one per question. No other text.
Example: ["What is Python?", "I'm interested in data science. What programming languages should I learn?", ...]"""


RAG_QUERY_KNOWLEDGE_BASE_PROMPT = """Based on the following knowledge base content, generate {batch_size} diverse factual questions that someone might ask to learn more about this topic.

Knowledge Base Content:
{knowledge_base_content}

Requirements for LENGTH VARIANCE (distribute evenly):
- SHORT (30%): 1 sentence, 5-15 words - Direct questions
  Example: "What is this about?", "How does it work?"
- MEDIUM (50%): 1-2 sentences, 15-40 words - Questions with some context
  Example: "I read about X in the content above. Can you explain how it relates to Y?"
- LONG (20%): 2-4 sentences, 40-80 words - Detailed questions with background
  Example: "I'm particularly interested in the section about X. I have some experience with Y but I'm not familiar with Z. Could you explain how these concepts connect and what I should focus on learning first?"

Additional requirements:
- Questions should be information-seeking and specific to the content
- Vary complexity from simple to advanced
- Include what, how, why, when, explain, describe type questions
- Escape quotes properly (use \\" for quotes inside strings)

Return ONLY a valid JSON array of strings, one per question. No other text.
Example: ["What is...?", "How do I...?", "I'm interested in X. Can you explain...?"]"""


# ==============================================================================
# STORE A CONTEXT PROMPTS
# ==============================================================================

CONTEXT_MESSAGE_PROMPT = """Generate {batch_size} diverse user messages for the "{category}" category.

Category Description: {description}

Examples:
{examples}

Requirements for LENGTH VARIANCE (distribute evenly):
- SHORT (25%): 1 sentence, 5-20 words - Brief statements
  Example: "I have 5 years of Python experience", "I'm feeling burned out at work"
- MEDIUM (50%): 2-3 sentences, 20-50 words - Moderate detail
  Example: "I have 5 years of Python experience and I'm certified in AWS. I've worked on both backend and data engineering projects."
- LONG (25%): 4-6 sentences, 50-120 words - Detailed narratives
  Example: "I've been working as a software engineer for 5 years, primarily with Python and Go. I started in backend development but transitioned to data engineering two years ago. I've worked on several large-scale ETL pipelines and have experience with AWS and GCP. Now I'm considering moving into a more specialized machine learning role, but I'm not sure if my current skill set is sufficient or what additional skills I should develop."

Additional requirements:
- Make them realistic and natural (as if typed by real users)
- Vary the tone (formal, casual, emotional, matter-of-fact)
- Use first-person statements ("I...", "My...", "I've...", "I'm...")
- Include varying levels of emotion and specificity
- Avoid repetitive patterns and templates
- Escape quotes properly (use \\" for quotes inside strings)

Return ONLY a valid JSON array of strings, one per message. No other text.
Example: ["I have 3 years of experience...", "I'm currently working as a data analyst. I really enjoy...", ...]"""


# ==============================================================================
# CHITCHAT MESSAGE PROMPTS
# ==============================================================================

CHITCHAT_MESSAGE_PROMPT = """Generate {batch_size} diverse small talk and social interaction messages for a chatbot conversation.

Include:
- Greetings (hi, hello, hey, good morning)
- Pleasantries (how are you, what's up, nice to meet you)
- Thank yous and acknowledgments (thanks, I appreciate it, got it)
- Short conversational responses (sounds good, that's great, I see)
- Friendly closings (have a great day, talk to you later)
- Light social interactions (how's it going, nice weather today)

Requirements for LENGTH VARIANCE (distribute evenly):
- VERY SHORT (50%): 1-3 words - Simple greetings and acknowledgments
  Example: "Hi!", "Thanks!", "Got it"
- SHORT (35%): 4-10 words - Standard social interactions
  Example: "How are you doing today?", "Thanks for your help!"
- MEDIUM (15%): 10-20 words - Conversational chitchat with a bit more substance
  Example: "Hey! How's it going? I hope you're having a great day so far."

Additional requirements:
- Natural and realistic social interactions
- Friendly and conversational tone
- NOT information-seeking questions about careers/topics
- NOT deep career-related statements
- Vary formality (casual to polite)
- Escape quotes properly (use \\" for quotes inside strings)

Return ONLY a valid JSON array of strings, one per message. No other text.
Example: ["Hi!", "How are you?", "Thanks so much!", "Hey! How's everything going with you?", ...]

IMPORTANT: Your response must be a valid JSON array starting with [ and ending with ]. Each message must be a string in quotes."""


# ==============================================================================
# OFF-TOPIC MESSAGE PROMPTS
# ==============================================================================
# Note: Off-topic messages are NOT used for training the BERT classifier.
# They are used for testing the semantic gate (Stage 1) which filters out
# off-topic messages before they reach the classifier (Stage 2).

OFFTOPIC_MESSAGE_PROMPT = """Generate {batch_size} diverse messages about completely random topics that have NOTHING to do with careers, jobs, professional development, learning, personal growth, or workplace topics.

Include messages about:
- Weather and nature (e.g., "It's really sunny today", "The mountains look beautiful")
- Sports and games (e.g., "Did you watch the football game?", "I love playing chess")
- Food and cooking (e.g., "I made pasta carbonara for dinner", "What's your favorite pizza topping?")
- Movies, TV shows, music (e.g., "That new Marvel movie was amazing", "I've been listening to jazz lately")
- Travel and places (e.g., "Paris in spring is magical", "I'm planning a trip to Japan")
- Hobbies and crafts (e.g., "I enjoy painting landscapes", "I'm learning to play guitar")
- Pets and animals (e.g., "My cat knocked over a plant", "Dogs are the best companions")
- Random facts (e.g., "Did you know elephants can't jump?", "The ocean is vast and mysterious")
- Daily life unrelated to work (e.g., "I need to buy groceries", "My neighbor is really loud")
- Technology/gadgets (non-career, e.g., "This new phone has a great camera", "I love my smart watch")
- General knowledge questions (e.g., "What's the capital of France?", "How do I fix a leaky faucet?")
- Health and fitness (non-work related, e.g., "I went for a 5-mile run today", "Yoga helps me relax")
- Family and relationships (e.g., "My sister is visiting this weekend", "I miss my grandparents")
- Entertainment and leisure (e.g., "I binge-watched a series last night", "Board games are fun")

Requirements for LENGTH VARIANCE (distribute evenly):
- SHORT (30%): 1 sentence, 5-15 words - Brief statements or questions
  Example: "What's the weather like?", "I love pizza!", "Did you see the game?"
- MEDIUM (50%): 2-3 sentences, 15-40 words - Moderate detail
  Example: "I went to the park yesterday and it was beautiful. The weather was perfect and there were lots of families having picnics."
- LONG (20%): 4-6 sentences, 40-100 words - Detailed narratives
  Example: "I've been thinking about getting a dog for a while now. I grew up with dogs and I really miss having a pet around the house. I've been researching different breeds and I think a golden retriever would be perfect for my lifestyle. They're friendly, active, and great with kids. I'm planning to visit some shelters this weekend to see if I can find the right match."

Additional requirements:
- Natural and conversational
- NO career, job, work, learning, or professional topics whatsoever
- NO emotional wellbeing or personal goals related to careers
- NO skills, education, or self-improvement in a professional context
- Vary between statements, questions, and exclamations
- Make them realistic things people might say
- Vary the tone (excited, casual, curious, storytelling)
- Escape quotes properly (use \\" for quotes inside strings)

Return ONLY a valid JSON array of strings, one per message. No other text.
Example: ["It's raining today", "I went to the beach yesterday and it was amazing. The water was so clear!", "I've been thinking about getting a cat..."]

IMPORTANT: Your response must be a valid JSON array starting with [ and ending with ]. Each message must be a string in quotes."""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def format_rag_query_generic(batch_size: int) -> str:
    """Format the generic RAG query prompt."""
    return RAG_QUERY_GENERIC_PROMPT.format(batch_size=batch_size)


def format_rag_query_knowledge_base(batch_size: int, knowledge_base_content: str) -> str:
    """Format the knowledge base RAG query prompt."""
    return RAG_QUERY_KNOWLEDGE_BASE_PROMPT.format(batch_size=batch_size, knowledge_base_content=knowledge_base_content)


def format_context_message(batch_size: int, category: str, description: str, examples: list) -> str:
    """Format the Store A context message prompt."""
    examples_text = "\n".join(f"- {ex}" for ex in examples)
    return CONTEXT_MESSAGE_PROMPT.format(batch_size=batch_size, category=category, description=description, examples=examples_text)


def format_chitchat_message(batch_size: int) -> str:
    """Format the chitchat message prompt."""
    return CHITCHAT_MESSAGE_PROMPT.format(batch_size=batch_size)


def format_offtopic_message(batch_size: int) -> str:
    """
    Format the off-topic message prompt.

    Note: Off-topic messages are used for testing the semantic gate (Stage 1),
    NOT for training the BERT classifier (Stage 2).
    """
    return OFFTOPIC_MESSAGE_PROMPT.format(batch_size=batch_size)
