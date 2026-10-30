"""ALL settings live here: folder paths, tunable numbers, and the NVIDIA key.

Every other file imports from this one. Nothing else reads .env or builds paths.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------- Folders ----------------
BASE_DIR = Path(__file__).resolve().parent.parent   # the rag_simple/ folder
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"     # uploaded files are saved here
QDRANT_DIR = BASE_DIR / "data" / "qdrant_storage"   # Qdrant's database folder
STATIC_DIR = BASE_DIR / "static"                    # the web page

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

# ---------------- Web server ----------------
HOST = "127.0.0.1"   # only your own computer can open it
PORT = 5000

# ---------------- Chunking ----------------
CHUNK_SIZE = 800     # max characters per chunk (about 200 tokens)
CHUNK_OVERLAP = 150  # characters repeated between neighbouring chunks

# ---------------- Embedding ----------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match the model above

# ---------------- Vector database ----------------
COLLECTION_NAME = "documents"
QDRANT_URL = os.getenv("QDRANT_URL", "")

# ---------------- Retrieval ----------------
TOP_K = 4              # how many closest chunks to fetch
SCORE_THRESHOLD = 0.30 # a chunk below this similarity is ignored

# ---------------- LLM (NVIDIA) ----------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
LLM_TEMPERATURE = 0.0   # 0 = factual and repeatable
LLM_MAX_TOKENS = 1024
# Extra model-specific settings. This model has a "thinking" mode; we switch it off.
# If NVIDIA rejects this field (e.g. you change models), set it to {}.
LLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
