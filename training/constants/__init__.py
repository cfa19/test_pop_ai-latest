# Training constants: categories, prompts, and prompt builders
# New hierarchical taxonomy: context > entity > sub_entity
# 5 contexts: professional, learning, social, psychological, personal
# + 3 non-context types: rag_query, chitchat, off_topic

from .categories import INTENT_CATEGORIES
from .prompts import (
    format_rag_query_generic,
    format_rag_query_knowledge_base,
    format_context_message,
    format_chitchat_message,
    format_offtopic_message,
    SYSTEM_MESSAGE,
)

# === New hierarchical taxonomy (5 contexts) ===
from .professional import (
    ENTITIES as PROFESSIONAL_ENTITIES,
    CONTEXT_NAME as PROFESSIONAL_CONTEXT_NAME,
    MULTI_LABEL_EXAMPLES as PROFESSIONAL_MULTI_LABEL_EXAMPLES,
    MESSAGE_GENERATION_SYSTEM_PROMPT as PROFESSIONAL_SYSTEM_PROMPT,
    build_message_generation_prompt as build_professional_prompt,
    build_multilabel_generation_prompt as build_professional_multilabel_prompt,
)
from .learning import (
    ENTITIES as LEARNING_ENTITIES,
    CONTEXT_NAME as LEARNING_CONTEXT_NAME,
    MULTI_LABEL_EXAMPLES as LEARNING_MULTI_LABEL_EXAMPLES,
    MESSAGE_GENERATION_SYSTEM_PROMPT as LEARNING_SYSTEM_PROMPT,
    build_message_generation_prompt as build_learning_prompt,
    build_multilabel_generation_prompt as build_learning_multilabel_prompt,
)
from .social import (
    ENTITIES as SOCIAL_ENTITIES,
    CONTEXT_NAME as SOCIAL_CONTEXT_NAME,
    MULTI_LABEL_EXAMPLES as SOCIAL_MULTI_LABEL_EXAMPLES,
    MESSAGE_GENERATION_SYSTEM_PROMPT as SOCIAL_SYSTEM_PROMPT,
    build_message_generation_prompt as build_social_prompt,
    build_multilabel_generation_prompt as build_social_multilabel_prompt,
)
from .psychological import (
    ENTITIES as PSYCHOLOGICAL_ENTITIES,
    CONTEXT_NAME as PSYCHOLOGICAL_CONTEXT_NAME,
    MULTI_LABEL_EXAMPLES as PSYCHOLOGICAL_MULTI_LABEL_EXAMPLES,
    MESSAGE_GENERATION_SYSTEM_PROMPT as PSYCHOLOGICAL_SYSTEM_PROMPT,
    build_message_generation_prompt as build_psychological_prompt,
    build_multilabel_generation_prompt as build_psychological_multilabel_prompt,
)
from .personal import (
    ENTITIES as PERSONAL_ENTITIES,
    CONTEXT_NAME as PERSONAL_CONTEXT_NAME,
    MULTI_LABEL_EXAMPLES as PERSONAL_MULTI_LABEL_EXAMPLES,
    MESSAGE_GENERATION_SYSTEM_PROMPT as PERSONAL_SYSTEM_PROMPT,
    build_message_generation_prompt as build_personal_prompt,
    build_multilabel_generation_prompt as build_personal_multilabel_prompt,
)

# === Non-context types (rag_query, chitchat, off_topic) ===
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

# Registry of all 5 contexts with their entities
CONTEXT_REGISTRY = {
    "professional": {
        "entities": PROFESSIONAL_ENTITIES,
        "system_prompt": PROFESSIONAL_SYSTEM_PROMPT,
        "build_prompt": build_professional_prompt,
        "build_multilabel_prompt": build_professional_multilabel_prompt,
        "multi_label_examples": PROFESSIONAL_MULTI_LABEL_EXAMPLES,
    },
    "learning": {
        "entities": LEARNING_ENTITIES,
        "system_prompt": LEARNING_SYSTEM_PROMPT,
        "build_prompt": build_learning_prompt,
        "build_multilabel_prompt": build_learning_multilabel_prompt,
        "multi_label_examples": LEARNING_MULTI_LABEL_EXAMPLES,
    },
    "social": {
        "entities": SOCIAL_ENTITIES,
        "system_prompt": SOCIAL_SYSTEM_PROMPT,
        "build_prompt": build_social_prompt,
        "build_multilabel_prompt": build_social_multilabel_prompt,
        "multi_label_examples": SOCIAL_MULTI_LABEL_EXAMPLES,
    },
    "psychological": {
        "entities": PSYCHOLOGICAL_ENTITIES,
        "system_prompt": PSYCHOLOGICAL_SYSTEM_PROMPT,
        "build_prompt": build_psychological_prompt,
        "build_multilabel_prompt": build_psychological_multilabel_prompt,
        "multi_label_examples": PSYCHOLOGICAL_MULTI_LABEL_EXAMPLES,
    },
    "personal": {
        "entities": PERSONAL_ENTITIES,
        "system_prompt": PERSONAL_SYSTEM_PROMPT,
        "build_prompt": build_personal_prompt,
        "build_multilabel_prompt": build_personal_multilabel_prompt,
        "multi_label_examples": PERSONAL_MULTI_LABEL_EXAMPLES,
    },
}

# Non-context types registry (flat structure, no entities)
NON_CONTEXT_REGISTRY = {
    "rag_query": {
        "categories": RAG_QUERY_CATEGORIES,
        "system_prompt": RAG_QUERY_SYSTEM_PROMPT,
        "build_prompt": build_rag_query_prompt,
    },
    "chitchat": {
        "categories": CHITCHAT_CATEGORIES,
        "system_prompt": CHITCHAT_SYSTEM_PROMPT,
        "build_prompt": build_chitchat_prompt,
    },
    "off_topic": {
        "categories": OFF_TOPIC_CATEGORIES,
        "system_prompt": OFF_TOPIC_SYSTEM_PROMPT,
        "build_prompt": build_off_topic_prompt,
    },
}

ALL_CONTEXTS = list(CONTEXT_REGISTRY.keys())
ALL_NON_CONTEXTS = list(NON_CONTEXT_REGISTRY.keys())
ALL_TYPES = ALL_CONTEXTS + ALL_NON_CONTEXTS

__all__ = [
    "INTENT_CATEGORIES",
    "format_rag_query_generic",
    "format_rag_query_knowledge_base",
    "format_context_message",
    "format_chitchat_message",
    "format_offtopic_message",
    "SYSTEM_MESSAGE",
    "CONTEXT_REGISTRY",
    "NON_CONTEXT_REGISTRY",
    "ALL_CONTEXTS",
    "ALL_NON_CONTEXTS",
    "ALL_TYPES",
    # Professional
    "PROFESSIONAL_ENTITIES",
    "PROFESSIONAL_SYSTEM_PROMPT",
    "build_professional_prompt",
    "build_professional_multilabel_prompt",
    # Learning
    "LEARNING_ENTITIES",
    "LEARNING_SYSTEM_PROMPT",
    "build_learning_prompt",
    "build_learning_multilabel_prompt",
    # Social
    "SOCIAL_ENTITIES",
    "SOCIAL_SYSTEM_PROMPT",
    "build_social_prompt",
    "build_social_multilabel_prompt",
    # Psychological
    "PSYCHOLOGICAL_ENTITIES",
    "PSYCHOLOGICAL_SYSTEM_PROMPT",
    "build_psychological_prompt",
    "build_psychological_multilabel_prompt",
    # Personal
    "PERSONAL_ENTITIES",
    "PERSONAL_SYSTEM_PROMPT",
    "build_personal_prompt",
    "build_personal_multilabel_prompt",
    # Non-context types
    "RAG_QUERY_CATEGORIES",
    "RAG_QUERY_SYSTEM_PROMPT",
    "build_rag_query_prompt",
    "CHITCHAT_CATEGORIES",
    "CHITCHAT_SYSTEM_PROMPT",
    "build_chitchat_prompt",
    "OFF_TOPIC_CATEGORIES",
    "OFF_TOPIC_SYSTEM_PROMPT",
    "build_off_topic_prompt",
]
