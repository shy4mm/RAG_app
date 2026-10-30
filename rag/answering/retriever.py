"""ANSWERING part 1: question -> the most relevant chunks."""
from rag import config
from rag.models import Hit


def retrieve(question: str, embedder, store) -> list[Hit]:
    question_vector = embedder.embed_query(question)       # 1. question -> numbers
    hits = store.search(question_vector, config.TOP_K)     # 2. find closest chunks
    for hit in hits:                                       # 3. keep only relevant ones
        hit.used = hit.score >= config.SCORE_THRESHOLD
    return hits
