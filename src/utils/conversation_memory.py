"""
Conversation Memory with RAG
Stores and retrieves conversation history using vector embeddings
"""

import logging
import uuid
from typing import Dict, List, Optional, Union

from fastapi import HTTPException
from openai import OpenAI
from supabase import Client
from voyageai.client import Client as VoyageAI

from src.config import RPCFunctions, Tables
from src.utils.rag import create_embedding


def store_message(supabase: Client, conversation_id: str, user_id: str, role: str, message: str, metadata: Optional[Dict] = None) -> str:
    """
    Store a message in conversation history WITHOUT an embedding
    Use this for messages that aren't worthy of indexing

    Args:
        supabase: Supabase client
        conversation_id: Unique conversation identifier
        user_id: User ID from authentication
        role: 'user' or 'assistant'
        message: The message text
        metadata: Optional metadata dict

    Returns:
        The ID of the stored message
    """
    try:
        # Store in conversation_history without embedding_id
        result = (
            supabase.table(Tables.CONVERSATION_HISTORY)
            .insert(
                {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "role": role,
                    "message": message,
                    "embedding_id": None,  # No embedding for this message
                    "metadata": metadata or {},
                }
            )
            .execute()
        )

        return result.data[0]["id"]

    except Exception:
        raise HTTPException(status_code=500, detail="An internal error occurred during message storage.")


def store_message_with_embedding(
    supabase: Client,
    embed_client: Union[OpenAI, VoyageAI],
    embed_model: str,
    conversation_id: str,
    user_id: str,
    role: str,
    message: str,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Store a message in conversation history with its embedding for RAG search.

    Args:
        supabase: Supabase client
        embed_client: Embedding client (OpenAI or VoyageAI)
        embed_model: Embedding model name
        conversation_id: Unique conversation identifier
        user_id: User ID from authentication
        role: 'user' or 'assistant'
        message: The message text
        metadata: Optional metadata dict

    Returns:
        The ID of the stored message
    """
    try:
        embedding = create_embedding(
            text=message,
            embed_client=embed_client,
            embed_model=embed_model,
        )

        embedding_result = (
            supabase.table(Tables.USER_EMBEDDINGS_1024)
            .insert({"user_id": user_id, "content": message, "embedding": embedding.embedding, "metadata": metadata or {}})
            .execute()
        )

        if not embedding_result.data:
            raise HTTPException(status_code=500, detail="Embedding insert returned no data.")
        embedding_id = embedding_result.data[0]["id"]

        result = (
            supabase.table(Tables.CONVERSATION_HISTORY)
            .insert(
                {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "role": role,
                    "message": message,
                    "embedding_id": embedding_id,
                    "metadata": metadata or {},
                }
            )
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=500, detail="Conversation history insert returned no data.")
        return result.data[0]["id"]

    except HTTPException:
        raise
    except Exception:
        logging.getLogger(__name__).exception("store_message_with_embedding failed")
        raise HTTPException(status_code=500, detail="An internal error occurred during message storage.")


def search_conversation_history(
    supabase: Client,
    embed_client: Union[OpenAI, VoyageAI],
    conversation_id: str,
    user_id: str,
    query: str,
    embed_model: str,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
) -> List[Dict]:
    """
    Search conversation history using RAG (vector similarity)

    Args:
        supabase: Supabase client
        embed_client: Embedding client (OpenAI or VoyageAI)
        conversation_id: Conversation to search in
        user_id: User ID for filtering
        query: Current user query to find relevant past messages
        embed_model: Embedding model name
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of relevant past messages with metadata
    """
    # Generate embedding for the query
    embedding = create_embedding(
        text=query,
        embed_client=embed_client,
        embed_model=embed_model,
    )
    query_embedding = embedding.embedding

    # Search using the database function
    result = supabase.rpc(
        RPCFunctions.SEARCH_CONVERSATION_HISTORY,
        {"query_embedding": query_embedding, "filter_conversation_id": conversation_id, "filter_user_id": user_id, "match_count": top_k},
    ).execute()

    # Filter by similarity threshold
    relevant_messages = [
        {"id": msg["id"], "role": msg["role"], "message": msg["message"], "similarity": msg["similarity"], "created_at": msg["created_at"]}
        for msg in result.data
        if msg["similarity"] >= similarity_threshold
    ]

    return relevant_messages


def generate_conversation_id() -> str:
    """
    Generate a unique conversation ID

    Returns:
        A UUID string
    """
    return str(uuid.uuid4())


def format_conversation_context(relevant_messages: List[Dict], include_recent: bool = True) -> str:
    """
    Format conversation history into a readable context string

    Args:
        relevant_messages: List of relevant messages from RAG search
        include_recent: Whether these are recent messages

    Returns:
        Formatted context string
    """
    if not relevant_messages:
        return ""

    context_lines = []

    if include_recent:
        context_lines.append("## Recent Conversation:")
    else:
        context_lines.append("## Relevant Past Conversation:")

    for msg in relevant_messages:
        role_label = "Usuario" if msg["role"] == "user" else "Asistente"
        context_lines.append(f"{role_label}: {msg['message']}")

    return "\n".join(context_lines)
