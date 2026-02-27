from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    embed_model: str | None = None
    chat_provider: Literal["openai", "groq", "voyage"] | None = None
    chat_model: str | None = None
    message_worth_method: str | None = None
    embed_provider: Literal["openai", "voyage"] | None = None
    # --- Disabled (token classifier handles off-topic via O label) ---
    # intent_classifier_type: Optional[Literal["openai", "bert"]] = None
    # semantic_gate_enabled: Optional[bool] = None


class IntentClassificationResponse(BaseModel):
    """Unified classification response (RAG query or Store A context)"""

    category: str  # rag_query, professional, psychological, learning, social, emotional, aspirational
    confidence: float
    secondary_categories: list[str] | None = None
    reasoning: str | None = None
    key_entities: dict | None = None


class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    sources: list[dict] | None = None
    conversation_context: list[dict] | None = None
    classification: IntentClassificationResponse | None = None
