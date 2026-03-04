"""
Embedding Models

Models for embedding generation using Voyage AI.
Default: voyage-3-large (1024 dimensions)
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    VOYAGE_3_LARGE = "voyage-3-large"
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_LITE = "voyage-3-lite"


class Embedding(BaseModel):
    """
    Response from embedding generation.

    Contains embedding vector of configurable dimensions.
    Supports any vector length (e.g., 512, 1024, 1536, 3072).
    """

    # ========================================================================
    # Output
    # ========================================================================
    embedding: list[float] = Field(..., min_length=1, description="Embedding vector of variable dimensions")

    # ========================================================================
    # Metadata
    # ========================================================================
    text: Optional[str] = Field(None, description="Original text that was embedded")
    model: str = Field(description="Model used for generation")
    dimension: int = Field(..., ge=1, description="Dimension of the embedding vector")
    tokens_used: Optional[int] = Field(None, ge=0, description="Number of tokens used")

    @model_validator(mode="after")
    def validate_embedding_length(self):
        """Validate that embedding length matches the dimension."""
        if len(self.embedding) != self.dimension:
            raise ValueError(f"Embedding length ({len(self.embedding)}) does not match dimension ({self.dimension})")
        return self
