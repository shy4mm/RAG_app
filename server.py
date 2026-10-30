"""The web server (FastAPI). It only translates web requests into pipeline calls.

Run:  python server.py
Then open  http://127.0.0.1:5000        (the chat page)
and        http://127.0.0.1:5000/docs   (FastAPI's automatic API page)
"""
import re
import shutil
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rag import config
from rag.pipeline import RagPipeline

app = FastAPI(title="Simple RAG")
pipeline = RagPipeline()

# Qdrant's local mode is not built for two requests at the same moment.
# FastAPI runs normal `def` endpoints in a thread pool, so we take turns with a lock.
lock = threading.Lock()


class AskRequest(BaseModel):
    question: str


def safe_name(filename: str) -> str:
    """Keep only the file name (no folders) and replace odd characters."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename or "").name)


@app.get("/")
def home():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/documents")
def list_documents():
    with lock:
        return pipeline.list_documents()


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...)):
    results = []
    with lock:
        for file in files:
            name = safe_name(file.filename)
            if Path(name).suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                results.append({"name": name or "?", "error": "unsupported file type"})
                continue
            path = config.DOCUMENTS_DIR / name
            with open(path, "wb") as out:
                shutil.copyfileobj(file.file, out)       # save the upload to disk
            try:
                results.append({"name": name, "chunks": pipeline.ingest_file(path)})
            except Exception as e:
                results.append({"name": name, "error": str(e)})
    return results


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    with lock:
        pipeline.delete_document(safe_name(name))
    return {"ok": True}


@app.post("/api/ask")
def ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        return JSONResponse({"error": "Type a question first."}, status_code=400)
    try:
        with lock:
            return pipeline.ask(question)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


if __name__ == "__main__":
    print(f"Open http://{config.HOST}:{config.PORT} in your browser")
    uvicorn.run(app, host=config.HOST, port=config.PORT)   # no reload: Qdrant allows one opener
