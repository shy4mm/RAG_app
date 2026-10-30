"""EMBEDDING: text -> list of 384 numbers that capture its meaning.

The SAME model is used for chunks and for questions. That is what makes
their vectors comparable.
"""
from sentence_transformers import SentenceTransformer

from rag import config


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.embed_texts([question])[0]
