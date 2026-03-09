import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

BASE_DIR = Path.cwd()
KNOWLEDGE_BASE = str(BASE_DIR / "Untitled Folder")
DB_NAME = str(BASE_DIR / "vector_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.0-flash"
RETRIEVAL_K = 10
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def get_google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata
        return userdata.get("GOOGLE_API_KEY")
    except Exception:
        return ""


def ensure_api_key():
    key = get_google_api_key()
    if key:
        os.environ["GOOGLE_API_KEY"] = key
