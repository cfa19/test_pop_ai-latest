"""
Runtime schemas for information extraction.

Moved from training/constants/ — these are runtime dependencies
(extraction prompts, NER patterns), not training code.
"""

from .info_extraction import (
    EXTRACTION_SCHEMAS,
    EXTRACTION_SYSTEM_MESSAGE,
    build_extraction_prompt,
)

__all__ = [
    "EXTRACTION_SCHEMAS",
    "EXTRACTION_SYSTEM_MESSAGE",
    "build_extraction_prompt",
]
