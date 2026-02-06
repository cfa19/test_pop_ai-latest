from typing import List

from src.config import Tables


def fetch_knowledge_base_content(supabase, limit: int = 50) -> List[str]:
    """
    Fetch content from general_embeddings_1024 to generate RAG queries.

    Args:
        supabase: Supabase client
        limit: Maximum number of documents to fetch

    Returns:
        List of content strings
    """
    print(f"Fetching up to {limit} documents from knowledge base...")

    try:
        result = supabase.table(Tables.GENERAL_EMBEDDINGS_1024).select("content").limit(limit).execute()

        contents = [row["content"] for row in result.data]
        print(f"Fetched {len(contents)} documents")
        return contents

    except Exception as e:
        print(f"Error fetching knowledge base: {e}")
        print("Will generate generic RAG queries without knowledge base context")
        return []
