"""The ONLY file that knows the order of the steps. Read this one to see the whole flow.

 UPLOAD:  file -> extract_text -> clean -> chunk -> embed -> save in Qdrant
 ASK:     question -> embed -> search Qdrant -> filter by threshold -> LLM -> answer
"""
from dataclasses import asdict
from pathlib import Path

from rag import config
from rag.answering.answerer import answer
from rag.answering.retriever import retrieve
from rag.database.vector_store import VectorStore
from rag.embedding.embedder import Embedder
from rag.ingestion.chunker import chunk_text
from rag.ingestion.extractor import clean_text, extract_text
from rag.models import Chunk


class RagPipeline:
    def __init__(self):
        print("Loading embedding model (first time downloads ~90 MB)...")
        self.embedder = Embedder()
        self.store = VectorStore()

    # ---------- upload side ----------
    def ingest_file(self, path: Path) -> int:
        text = clean_text(extract_text(path))                       # ingestion
        pieces = chunk_text(text)                                   # ingestion
        if not pieces:
            return 0
        chunks = [Chunk(t, path.name, i) for i, t in enumerate(pieces)]
        vectors = self.embedder.embed_texts([c.text for c in chunks])  # embedding
        self.store.delete_source(path.name)                         # database (no duplicates)
        self.store.add_chunks(chunks, vectors)                      # database
        return len(chunks)

    def delete_document(self, name: str) -> None:
        self.store.delete_source(name)
        (config.DOCUMENTS_DIR / name).unlink(missing_ok=True)

    def list_documents(self) -> list[dict]:
        docs = []
        for p in sorted(config.DOCUMENTS_DIR.iterdir()):
            if p.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                docs.append({"name": p.name,
                             "size_kb": round(p.stat().st_size / 1024, 1),
                             "chunks": self.store.count_for_source(p.name)})
        return docs

    # ---------- question side ----------
    def ask(self, question: str) -> dict:
        hits = retrieve(question, self.embedder, self.store)        # answering: retriever
        text, messages = answer(question, hits)                     # answering: answerer
        return {"answer": text,
                "chunks": [asdict(h) for h in hits],
                "prompt": messages}
