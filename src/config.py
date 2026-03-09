import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# Project root (parent of src/) so paths work when running "python src/app.py" from repo root
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = str(BASE_DIR / "PDFs")
# print(KNOWLEDGE_BASE)
# os.listdir(KNOWLEDGE_BASE)
DB_NAME = str(BASE_DIR / "vector_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
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
