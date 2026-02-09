"""
Pydantic Models Package

This package contains all Pydantic models for the Activity Harmonia platform.

Subpackages:
- store_a: Canonical Profile models (Store A)
- store_b: Journal/Event Sourcing models (Store B)
"""

from .embeddings import Embedding

__all__ = [
    "Embedding",
]

__version__ = "1.0.0"
