import faiss
import numpy as np
import pickle
import os


FAISS_INDEX_PATH = "models/faiss_index.bin"
CHUNKS_PATH = "models/chunks.pkl"


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index from embeddings.
    Uses IndexFlatL2 — exact search, no approximation.
    """
    embedding_dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension: {embedding_dim}")

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.astype(np.float32))

    print(f"FAISS index built. Total vectors stored: {index.ntotal}")
    return index


def save_index(index: faiss.Index, chunks: list[dict]):
    """
    Saves FAISS index and chunks to disk so you
    don't need to re-embed every time you restart.
    """
    os.makedirs("models", exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {CHUNKS_PATH}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    """
    Loads FAISS index and chunks from disk.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Run build first.")

    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"FAISS index loaded. Total vectors: {index.ntotal}")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"Chunks loaded. Total chunks: {len(chunks)}")

    return index, chunks


def search_faiss(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = 10
) -> list[dict]:
    """
    Searches FAISS index with a query embedding.
    Returns top_k most similar chunks with their scores.
    """
    query = query_embedding.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # -1 means no result found
            chunk = chunks[idx].copy()
            chunk["score"] = float(dist)
            chunk["retrieval_type"] = "semantic"
            results.append(chunk)

    return results


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.document_loader import load_pdfs_from_folder
    from src.text_chunker import chunk_documents
    from src.embedder import load_embedding_model, generate_embeddings

    # Build and save index
    pages = load_pdfs_from_folder("data/")
    chunks = chunk_documents(pages)
    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    # Test search
    print("\n--- Testing Search ---")
    test_query = "What is the attention mechanism?"
    query_embedding = model.encode([test_query], convert_to_numpy=True)
    results = search_faiss(query_embedding, index, chunks, top_k=5)

    print(f"\nQuery: {test_query}")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Source: {r['source']} | Page: {r['page']}")
        print(f"  Score: {r['score']:.4f}")
        print(f"  Preview: {r['text'][:200]}...")
        print()
