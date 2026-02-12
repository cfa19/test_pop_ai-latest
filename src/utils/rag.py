import random
import time
from typing import List, Union

from openai import APIError, OpenAI, RateLimitError
from voyageai.client import Client as VoyageAI

from src.config import RRF_K, RPCFunctions, get_supabase
from src.models import DocumentChunk, Embedding


# ===============================
# CHUNK PARSING FUNCTIONS
# ===============================
def parse_chunks_file(file_path: str) -> List[DocumentChunk]:
    """
    Parse a pre-chunked markdown file and extract chunks with metadata.

    Expected format:
    ## CHUNK X: Title

    **Section:** ...
    **Subsection:** ... (optional)
    **Chunk ID:** X

    Content here...

    ---

    Args:
        file_path: Path to the chunks .md file

    Returns:
        List of DocumentChunk objects with content and metadata
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return []

    chunks = []
    # Split by chunk headers
    chunk_sections = content.split("## CHUNK")

    for section in chunk_sections[1:]:  # Skip first empty part before first chunk
        lines = section.split("\n")
        if not lines:
            continue

        # Parse header: X: Title
        header_line = lines[0].strip()
        if ":" not in header_line:
            continue

        parts = header_line.split(":", 1)
        try:
            chunk_id = int(parts[0].strip())
        except ValueError:
            continue

        title = parts[1].strip() if len(parts) > 1 else ""

        # Initialize metadata
        metadata = {"chunk_id": chunk_id, "title": title, "section": None, "subsection": None}

        # Parse metadata and content
        content_lines = []
        metadata_done = False

        for i, line in enumerate(lines[1:], 1):
            line_stripped = line.strip()

            if not metadata_done:
                # Still in metadata section
                if line_stripped.startswith("**Section:**"):
                    metadata["section"] = line_stripped.replace("**Section:**", "").strip()
                elif line_stripped.startswith("**Subsection:**"):
                    metadata["subsection"] = line_stripped.replace("**Subsection:**", "").strip()
                elif line_stripped.startswith("**Chunk ID:**"):
                    # Update chunk_id if specified
                    try:
                        chunk_id_from_meta = int(line_stripped.replace("**Chunk ID:**", "").strip())
                        metadata["chunk_id"] = chunk_id_from_meta
                    except ValueError:
                        pass
                elif line_stripped == "":
                    # Empty line - might be end of metadata or just spacing
                    continue
                elif line_stripped == "---":
                    # Separator - end of metadata, start content collection
                    metadata_done = True
                    continue
                else:
                    # First non-metadata line - start collecting content
                    metadata_done = True
                    if line_stripped:  # Only add non-empty lines
                        content_lines.append(line.rstrip("\n"))
            else:
                # Collecting content
                if line_stripped == "---":
                    # End of chunk
                    break
                else:
                    content_lines.append(line.rstrip("\n"))

        # Create DocumentChunk object
        content = "\n".join(content_lines).strip()
        if content:  # Only add chunks with content
            # Remove None values from metadata
            metadata_clean = {k: v for k, v in metadata.items() if v is not None}
            chunk = DocumentChunk(content=content, metadata=metadata_clean if metadata_clean else None)
            chunks.append(chunk)

    return chunks


# ===============================
# EMBEDDING FUNCTIONS
# ===============================
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


def create_embeddings_batch(
    chunks: List[DocumentChunk],
    embed_client: OpenAI,
    embed_model: str,
    embed_dimensions: int,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
) -> List[Embedding]:
    """
    Generate embeddings in batch with retries.
    Voyage AI allows up to 128 texts per request.

    Args:
        chunks: List of chunks
        max_retries: Maximum number of retries
        initial_backoff: Initial wait time

    Returns:
        List of Embedding objects with dimensions matching the returned embeddings
    """
    backoff = initial_backoff

    for attempt in range(max_retries):
        try:
            response = embed_client.embed(
                model=embed_model,
                texts=[chunk.content for chunk in chunks],
            )
            # Sort by index to maintain original order
            embeddings = [None] * len(chunks)
            for i, embedding in enumerate(response.embeddings):
                embeddings[i] = Embedding(
                    embedding=embedding, model=embed_model, dimension=len(embedding), tokens_used=getattr(response, "total_tokens", None)
                )
            return embeddings

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e

            sleep_time = min(backoff * (2**attempt) + random.uniform(0, 1), max_backoff)
            print(f"      Rate limit in batch, waiting {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(sleep_time)

        except APIError as e:
            if attempt == max_retries - 1:
                raise e

            sleep_time = min(backoff * (2**attempt), max_backoff)
            print(f"      API error in batch, retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(sleep_time)

    raise Exception(f"Could not generate batch embeddings after {max_retries} attempts")


# ===============================
# SEARCH FUNCTIONS
# ===============================
def semantic_search(query: str, embed_client: OpenAI, embed_model: str, embed_dimensions: int, top_k: int = 5):
    """Search only by embedding similarity"""
    query_embedding = create_embedding(query, embed_client, embed_model)

    result = get_supabase().rpc(RPCFunctions.RAG_SEARCH_SEMANTIC, {"query_embedding": query_embedding.embedding, "match_count": top_k}).execute()

    return result.data


def fulltext_search(query: str, top_k: int = 5):
    """Search only by keywords"""
    result = get_supabase().rpc(RPCFunctions.RAG_SEARCH_FULLTEXT, {"query_text": query, "match_count": top_k}).execute()

    return result.data


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
