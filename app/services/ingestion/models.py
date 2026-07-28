from dataclasses import dataclass, field
from typing import Any, Dict
import uuid

@dataclass
class ExtractedPage:
    """Represents raw text extracted from a single page or section of a document."""
    text: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedChunk:
    """Represents a token-bounded chunk ready for embedding and vector storage."""
    text: str                                                      # No default
    token_count: int                                               # No default
    id: str = field(default_factory=lambda: str(uuid.uuid4()))    # Default factory
    metadata: Dict[str, Any] = field(default_factory=dict)         # Default factory