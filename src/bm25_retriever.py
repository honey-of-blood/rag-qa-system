import pickle
import os
from rank_bm25 import BM25Okapi


BM25_PATH = "models/bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.
    BM25 works on token lists not raw strings.
    """
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Builds a BM25 index from all chunk texts.
    """
    print(f"Building BM25 index for {len(chunks)} chunks...")
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index built successfully")
    return bm25


def save_bm25_index(bm25: BM25Okapi):
    """
    Saves BM25 index to disk.
    """
    os.makedirs("models", exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {BM25_PATH}")


def load_bm25_index() -> BM25Okapi:
    """
    Loads BM25 index from disk.
    """
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError("BM25 index not found. Build it first.")
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print("BM25 index loaded successfully")
    return bm25


def search_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 10
) -> list[dict]:
    """
    Searches BM25 index with a raw text query.
    Returns top_k most relevant chunks with scores.
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top_k indices sorted by score descending
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # skip zero score results
            chunk = chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["retrieval_type"] = "keyword"
            results.append(chunk)

    return results


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.document_loader import load_pdfs_from_folder
    from src.text_chunker import chunk_documents
    from src.vector_store import load_index

    # Load chunks from saved index
    _, chunks = load_index()

    # Build and save BM25
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    # Test search
    test_query = "attention mechanism transformer"
    print(f"\n--- BM25 Search Test ---")
    print(f"Query: '{test_query}'\n")

    results = search_bm25(test_query, bm25, chunks, top_k=5)
    for i, r in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Source: {r['source']} | Page: {r['page']}")
        print(f"  BM25 Score: {r['score']:.4f}")
        print(f"  Preview: {r['text'][:200]}...")
        print()
