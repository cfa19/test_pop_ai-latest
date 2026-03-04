"""
User Context Search

Search worthy messages in retrieval_chunks for user profile RAG.
Reuses create_embedding() from rag.py for query embedding generation.
"""

from typing import Dict, List

from openai import OpenAI
from supabase import Client

from src.config import RPCFunctions, SourceTypes
from src.utils.rag import create_embedding


def search_user_context(
    supabase: Client,
    embed_client: OpenAI,
    user_id: str,
    query: str,
    embed_model: str,
    embed_dimensions: int,
    top_k: int = 5,
    min_confidence: float = 0.6,
) -> List[Dict]:
    """
    Search user's worthy messages in retrieval_chunks.

    Only searches high-value indexed messages, not all conversation history.

    Args:
        supabase: Supabase client
        embed_client: Embedding client (Voyage AI via OpenAI SDK)
        user_id: User ID to filter by
        query: Search query text
        embed_model: Embedding model name
        embed_dimensions: Embedding dimensions
        top_k: Number of results to return
        min_confidence: Minimum confidence score for results

    Returns:
        List of matching chunks with similarity scores
    """
    # Generate query embedding reusing existing function
    query_emb = create_embedding(query, embed_client, embed_model)

    # Search retrieval_chunks via Supabase RPC
    result = supabase.rpc(
        RPCFunctions.SEARCH_USER_CONTEXT_CHUNKS,
        {
            "query_embedding": query_emb.embedding,
            "filter_user_id": user_id,
            "filter_source_type": SourceTypes.CONVERSATION_MESSAGE,
            "match_count": top_k,
            "min_confidence": min_confidence,
        },
    ).execute()

    return result.data
