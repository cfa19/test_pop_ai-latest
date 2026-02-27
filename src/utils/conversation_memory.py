"""
Conversation Memory with RAG
Stores and retrieves conversation history using vector embeddings
"""

import logging
import uuid

from fastapi import HTTPException
from openai import OpenAI
from supabase import Client
from voyageai.client import Client as VoyageAI

from src.config import RPCFunctions, Tables
from src.utils.rag import create_embedding


def store_message(supabase: Client, conversation_id: str, user_id: str, role: str, message: str, metadata: dict | None = None) -> str:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred during message storage.") from e


def store_message_with_embedding(
    supabase: Client,
    embed_client: OpenAI | VoyageAI,
    embed_model: str,
    conversation_id: str,
    user_id: str,
    role: str,
    message: str,
    metadata: dict | None = None,
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
    except Exception as e:
        logging.getLogger(__name__).exception("store_message_with_embedding failed")
        raise HTTPException(status_code=500, detail="An internal error occurred during message storage.") from e


def search_conversation_history(
    supabase: Client,
    embed_client: OpenAI | VoyageAI,
    conversation_id: str,
    user_id: str,
    query: str,
    embed_model: str,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
) -> list[dict]:
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
    return [
        {"id": msg["id"], "role": msg["role"], "message": msg["message"], "similarity": msg["similarity"], "created_at": msg["created_at"]}
        for msg in result.data
        if msg["similarity"] >= similarity_threshold
    ]


def check_duplicate_message(
    supabase: Client,
    user_id: str,
    message: str,
) -> bool:
    """
    Check if this exact message was already sent by this user.

    Queries conversation_history for matching text. If count > 1,
    the message was already processed (current + previous = duplicate).

    Args:
        supabase: Supabase client
        user_id: User ID
        message: Message text to check

    Returns:
        True if a duplicate was found, False otherwise
    """
    logger = logging.getLogger(__name__)
    try:
        result = (
            supabase.table(Tables.CONVERSATION_HISTORY)
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("role", "user")
            .eq("message", message)
            .execute()
        )
        count = result.count or 0
        if count > 1:
            logger.info(f"[DEDUP] Found {count} copies for user {user_id[:8]}...")
        return count > 1
    except Exception:
        logger.exception("[DEDUP] Check failed, proceeding with extraction")
        return False


def get_recent_messages(supabase: Client, conversation_id: str, user_id: str, limit: int = 5) -> list[dict]:
    """
    Get the most recent messages from a conversation (for context window)

    Args:
        supabase: Supabase client
        conversation_id: Conversation ID
        user_id: User ID for filtering
        limit: Number of recent messages to retrieve

    Returns:
        List of recent messages in chronological order
    """
    result = (
        supabase.table(Tables.CONVERSATION_HISTORY)
        .select("id, role, message, created_at")
        .eq("conversation_id", conversation_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    # Reverse to get chronological order (oldest to newest)
    return list(reversed(result.data))


def generate_conversation_id() -> str:
    """
    Generate a unique conversation ID

    Returns:
        A UUID string
    """
    return str(uuid.uuid4())


def format_conversation_context(relevant_messages: list[dict], include_recent: bool = True) -> str:
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
