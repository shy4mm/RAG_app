# Simple RAG (single user, runs on localhost)

## Run
    python3 -m venv .venv && source .venv/bin/activate
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    pip install --no-cache-dir -r requirements.txt
    cp .env.example .env        # put your NVIDIA key inside .env
    python server.py            # open http://127.0.0.1:5000

## Where to read the code (in this order)
1. rag/config.py            all settings and paths
2. rag/pipeline.py          the whole flow in ~60 lines
3. rag/ingestion/           extractor.py (file -> text), chunker.py (text -> chunks)
4. rag/embedding/           embedder.py (text -> numbers)
5. rag/database/            vector_store.py (Qdrant)
6. rag/answering/           retriever.py -> prompt.py -> llm_client.py -> answerer.py
7. server.py + static/index.html   the web part (FastAPI; auto docs at /docs)
# RAG_app
