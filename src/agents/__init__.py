"""
Agents Package

LangGraph workflow for intelligent message processing:
- Intent Classification: Classifies into 9 categories (rag_query + 6 Store A contexts + chitchat + off_topic)
- RAG Retrieval: Searches documents and conversation history
- Response Generation: Context-specific personalized responses
"""

from .langgraph_workflow import IntentClassification, MessageCategory, WorkflowState, create_workflow, run_workflow

__all__ = [
    "run_workflow",
    "create_workflow",
    "WorkflowState",
    "IntentClassification",
    "MessageCategory",
]
