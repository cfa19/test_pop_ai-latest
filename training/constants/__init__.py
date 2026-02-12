# Training constants: categories, prompts, and prompt builders
from .categories import INTENT_CATEGORIES
from .prompts import (
    format_rag_query_generic,
    format_rag_query_knowledge_base,
    format_context_message,
    format_chitchat_message,
    format_offtopic_message,
    SYSTEM_MESSAGE,
)
from .aspirational import (
    ASPIRATION_CATEGORIES,
    build_message_generation_prompt as build_aspirational_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as ASPIRATIONAL_SYSTEM_PROMPT,
)
from .professional import (
    PROFESSIONAL_CATEGORIES,
    build_message_generation_prompt as build_professional_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as PROFESSIONAL_SYSTEM_PROMPT,
)
from .psychological import (
    PSYCHOLOGICAL_CATEGORIES,
    build_message_generation_prompt as build_psychological_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as PSYCHOLOGICAL_SYSTEM_PROMPT,
)
from .learning import (
    LEARNING_CATEGORIES,
    build_message_generation_prompt as build_learning_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as LEARNING_SYSTEM_PROMPT,
)
from .social import (
    SOCIAL_CATEGORIES,
    build_message_generation_prompt as build_social_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as SOCIAL_SYSTEM_PROMPT,
)
from .emotional import (
    EMOTIONAL_CATEGORIES,
    build_message_generation_prompt as build_emotional_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as EMOTIONAL_SYSTEM_PROMPT,
)
from .rag_query import (
    RAG_QUERY_CATEGORIES,
    build_message_generation_prompt as build_rag_query_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as RAG_QUERY_SYSTEM_PROMPT,
)
from .chitchat import (
    CHITCHAT_CATEGORIES,
    build_message_generation_prompt as build_chitchat_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as CHITCHAT_SYSTEM_PROMPT,
)
from .off_topic import (
    OFF_TOPIC_CATEGORIES,
    build_message_generation_prompt as build_off_topic_prompt,
    MESSAGE_GENERATION_SYSTEM_PROMPT as OFF_TOPIC_SYSTEM_PROMPT,
)

__all__ = [
    "INTENT_CATEGORIES",
    "format_rag_query_generic",
    "format_rag_query_knowledge_base",
    "format_context_message",
    "format_chitchat_message",
    "format_offtopic_message",
    "SYSTEM_MESSAGE",
    "ASPIRATION_CATEGORIES",
    "build_aspirational_prompt",
    "ASPIRATIONAL_SYSTEM_PROMPT",
    "PROFESSIONAL_CATEGORIES",
    "build_professional_prompt",
    "PROFESSIONAL_SYSTEM_PROMPT",
    "PSYCHOLOGICAL_CATEGORIES",
    "build_psychological_prompt",
    "PSYCHOLOGICAL_SYSTEM_PROMPT",
    "LEARNING_CATEGORIES",
    "build_learning_prompt",
    "LEARNING_SYSTEM_PROMPT",
    "SOCIAL_CATEGORIES",
    "build_social_prompt",
    "SOCIAL_SYSTEM_PROMPT",
    "EMOTIONAL_CATEGORIES",
    "build_emotional_prompt",
    "EMOTIONAL_SYSTEM_PROMPT",
    "RAG_QUERY_CATEGORIES",
    "build_rag_query_prompt",
    "RAG_QUERY_SYSTEM_PROMPT",
    "CHITCHAT_CATEGORIES",
    "build_chitchat_prompt",
    "CHITCHAT_SYSTEM_PROMPT",
    "OFF_TOPIC_CATEGORIES",
    "build_off_topic_prompt",
    "OFF_TOPIC_SYSTEM_PROMPT",
]
