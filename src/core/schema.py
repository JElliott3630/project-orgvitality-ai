from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Chunk:
    id: str
    text: str
    source: str          # file name
    location: str        # "page 4", "slide 2", etc.
    start_token: int
    embedding: Optional[List[float]] = None

@dataclass
class DocumentBatch:
    doc_id: str
    chunks: List[Chunk]
