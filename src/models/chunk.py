"""
Document Chunk Models

General models for document chunks used in RAG systems.
"""

from typing import Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Document chunk for RAG ingestion.

    Used when loading pre-chunked documents.
    """

    # ========================================================================
    # Content
    # ========================================================================
    content: str = Field(min_length=1, description="Text content of the chunk")

    # ========================================================================
    # Metadata
    # ========================================================================
    metadata: Optional[dict] = Field(None, description="Additional metadata for the chunk (section, subsection, chunk_id, etc.)")
