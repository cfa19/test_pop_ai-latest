from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    semantic_gate_enabled: Optional[bool] = None


class IntentClassificationResponse(BaseModel):
    """Unified classification response (RAG query or Store A context)"""

    category: str  # rag_query, professional, psychological, learning, social, emotional, aspirational
    confidence: float
    secondary_categories: Optional[list[str]] = None
    reasoning: Optional[str] = None
    key_entities: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    sources: Optional[list[dict]] = None
    conversation_context: Optional[list[dict]] = None
    classification: Optional[IntentClassificationResponse] = None
