import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chromadb"
LOG_DIR = DATA_DIR / "logs"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_questions.json"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"

# Ingestion origin
REPO_OWNER = os.getenv("REPO_OWNER", "evidentlyai")
REPO_NAME = os.getenv("REPO_NAME", "docs")

# RAG
COLLECTION_NAME = "evidently_docs"
EMBEDDING_MODEL_NAME = "multi-qa-distilbert-cos-v1"
CHUNK_SIZE = 2000
CHUNK_STEP = 1000

# Models
AGENT_MODEL = "openai:gpt-4o-mini"
EVAL_MODEL = "openai:gpt-5-nano"


def load_env():
    """Load environment variables from .env file."""
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment")
