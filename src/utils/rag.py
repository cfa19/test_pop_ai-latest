from typing import Union

from openai import OpenAI
from voyageai.client import Client as VoyageAI

from src.config import RRF_K, RPCFunctions, get_supabase
from src.models import Embedding


def create_embedding(text: str, embed_client: Union[OpenAI, VoyageAI], embed_model: str) -> Embedding:
    """Generate embedding with Voyage AI. Returns an Embedding with text and vector."""
    if isinstance(embed_client, OpenAI):
        response = embed_client.embeddings.create(
            model=embed_model,
            input=text,
        )
        if not response.data:
            raise ValueError("OpenAI embedding returned no data")
        item = response.data[0]
        vec = item.embedding
        usage = getattr(response, "usage", None)
        tokens_used = usage.total_tokens if usage else None
    elif isinstance(embed_client, VoyageAI):
        response = embed_client.embed(
            model=embed_model,
            texts=[text],
        )
        vec = response.embeddings[0]
        tokens_used = getattr(response, "tokens_used", None)
    else:
        raise ValueError(f"Unsupported embed client: {type(embed_client)}. Must be OpenAI or VoyageAI.")

    return Embedding(text=text, embedding=vec, model=embed_model, dimension=len(vec), tokens_used=tokens_used)


def hybrid_search(query: str, embed_client: OpenAI, embed_model: str, embed_dimensions: int, top_k: int = 5, semantic_weight: float = 0.5):
    """
    Hybrid search combining semantic + full-text with RRF

    Args:
        query: Search text
        top_k: Number of results
        semantic_weight: Semantic search weight (0.0 to 1.0)
            - 1.0 = 100% semantic (embeddings only)
            - 0.0 = 100% full-text (keywords only)
            - 0.5 = 50/50 (balanced, recommended)
    """
    query_embedding = create_embedding(query, embed_client, embed_model)

    result = (
        get_supabase()
        .rpc(
            RPCFunctions.RAG_HYBRID_SEARCH,
            {
                "query_text": query,
                "query_embedding": query_embedding.embedding,
                "match_count": top_k,
                "semantic_weight": semantic_weight,
                "rrf_k": RRF_K,
            },
        )
        .execute()
    )

    return result.data
