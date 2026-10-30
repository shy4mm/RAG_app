"""The two small data shapes that travel between modules."""
from dataclasses import dataclass


@dataclass
class Chunk:
    """A piece of a document (made by ingestion, stored by database)."""
    text: str
    source: str       # file name
    chunk_index: int  # 0, 1, 2 ... position inside that file


@dataclass
class Hit:
    """A chunk found by a search (made by database, judged by retriever)."""
    text: str
    source: str
    chunk_index: int
    score: float          # cosine similarity: 1.0 = same meaning, 0 = unrelated
    used: bool = False    # True if it passed the threshold and goes to the LLM
