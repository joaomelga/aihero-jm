"""
Search module: vector search over ChromaDB collection.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

_collection = None
_embedding_model = None


def get_collection() -> chromadb.Collection:
    """Get the ChromaDB collection (cached)."""
    global _collection

    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)

    return _collection


def get_embedding_model() -> SentenceTransformer:
    """Get the embedding model (cached)."""
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def vector_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Perform vector similarity search on the ChromaDB collection.

    Returns a list of dicts with keys: chunk, filename, title, description, start.
    """
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results,
    )

    output = []
    for i in range(len(results["ids"][0])):
        item = {
            "chunk": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "title": results["metadatas"][0][i].get("title", ""),
            "description": results["metadatas"][0][i].get("description", ""),
            "start": results["metadatas"][0][i].get("start", 0),
        }
        output.append(item)

    return output
