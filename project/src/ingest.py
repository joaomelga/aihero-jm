"""
Ingestion pipeline: download repo → chunk → embed → store in ChromaDB.

Usage:
    python -m src.ingest
"""

import io
import zipfile

import chromadb
import frontmatter
import requests
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.config import (
    CHUNK_SIZE,
    CHUNK_STEP,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    REPO_NAME,
    REPO_OWNER,
    load_env,
)


def read_repo_data(repo_owner: str, repo_name: str) -> list[dict]:
    """Download and parse all markdown files from a GitHub repository."""
    prefix = "https://codeload.github.com"
    url = f"{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main"
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()

        if not (filename_lower.endswith(".md") or filename_lower.endswith(".mdx")):
            continue

        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode("utf-8", errors="ignore")
                post = frontmatter.loads(content)
                data = post.to_dict()
                data["filename"] = filename
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    zf.close()
    return repository_data


def sliding_window(seq: str, size: int, step: int) -> list[dict]:
    """Split a string into overlapping chunks using a sliding window."""
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []

    for i in range(0, n, step):
        chunk = seq[i : i + size]
        result.append({"start": i, "chunk": chunk})

        if i + size >= n:
            break

    return result


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Chunk all documents using sliding window."""
    all_chunks = []
    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        chunks = sliding_window(doc_content, CHUNK_SIZE, CHUNK_STEP)

        for chunk in chunks:
            chunk.update(doc_copy)

        all_chunks.extend(chunks)

    return all_chunks


def embed_and_store(chunks: list[dict]) -> int:
    """Embed chunks and upsert into ChromaDB. Returns the collection count."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting"):
        batch = chunks[i : i + batch_size]

        ids = [f"{c['filename']}_{c['start']}" for c in batch]
        documents = [c["chunk"] for c in batch]
        embeddings = model.encode(documents).tolist()
        metadatas = [
            {
                "filename": c["filename"],
                "title": c.get("title", ""),
                "description": c.get("description", ""),
                "start": c["start"],
            }
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    return collection.count()


if __name__ == "__main__":
    load_env()

    print(f"Downloading {REPO_OWNER}/{REPO_NAME}...")
    docs = read_repo_data(REPO_OWNER, REPO_NAME)
    print(f"Downloaded {len(docs)} markdown files")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Embedding and storing in ChromaDB...")
    count = embed_and_store(chunks)
    print(f"ChromaDB collection has {count} documents")
    print(f"Persisted to {CHROMA_PERSIST_DIR}")
