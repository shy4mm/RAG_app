"""DATABASE: save and search vectors in Qdrant.

Two modes, chosen by QDRANT_URL in .env:
  - QDRANT_URL empty            -> embedded mode (a folder on disk, no server, no dashboard)
  - QDRANT_URL=http://localhost:6333 -> Qdrant server in Docker (dashboard at /dashboard)
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector,
                                  MatchValue, PointStruct, VectorParams)

from rag import config
from rag.models import Chunk, Hit


def _source_filter(source: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


class VectorStore:
    def __init__(self):
        if config.QDRANT_URL:
            self.client = QdrantClient(url=config.QDRANT_URL)          # Docker server
            print(f"Qdrant: using server at {config.QDRANT_URL}")
        else:
            self.client = QdrantClient(path=str(config.QDRANT_DIR))    # embedded folder
            print(f"Qdrant: using local folder {config.QDRANT_DIR}")

        if not self.client.collection_exists(config.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Each point = a vector + a payload (the text and where it came from)."""
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.source}:{c.chunk_index}")),
                vector=vec,
                payload={"text": c.text, "source": c.source, "chunk_index": c.chunk_index},
            )
            for c, vec in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=config.COLLECTION_NAME, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[Hit]:
        """Return the top_k closest chunks with their similarity score."""
        result = self.client.query_points(
            collection_name=config.COLLECTION_NAME, query=query_vector, limit=top_k)
        return [Hit(p.payload["text"], p.payload["source"], p.payload["chunk_index"], p.score)
                for p in result.points]

    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from one file."""
        self.client.delete(collection_name=config.COLLECTION_NAME,
                           points_selector=FilterSelector(filter=_source_filter(source)))

    def count_for_source(self, source: str) -> int:
        return self.client.count(collection_name=config.COLLECTION_NAME,
                                 count_filter=_source_filter(source), exact=True).count