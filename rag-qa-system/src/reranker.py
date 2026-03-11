from sentence_transformers import CrossEncoder
import time


# Free, small, fast cross-encoder model (~85MB, runs on CPU)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_model = None


def load_reranker():
    """
    Loads the cross-encoder reranker model.
    Uses a global singleton so it only loads once.
    Downloads ~85MB on first run.
    """
    global _reranker_model
    if _reranker_model is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        print("Reranker model loaded successfully")
    return _reranker_model


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Takes a query and a list of candidate chunks.
    Scores each chunk against the query using cross-encoder.
    Returns top_k chunks sorted by relevance score descending.
    """
    if not chunks:
        return []

    reranker = load_reranker()

    # Build (query, chunk_text) pairs for cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs
    start = time.time()
    scores = reranker.predict(pairs)
    elapsed = time.time() - start

    print(f"Reranking {len(chunks)} chunks took {elapsed:.2f}s")

    # Attach scores to chunks
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = float(score)
        scored_chunks.append(chunk_copy)

    # Sort by rerank score descending
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return top_k
    return scored_chunks[:top_k]


def rerank_and_compare(
    query: str,
    chunks: list[dict],
    top_k: int = 5
) -> dict:
    """
    Returns both before and after reranking so you can
    compare the ordering change — useful for evaluation.
    """
    before = chunks[:top_k]
    after = rerank_chunks(query, chunks, top_k=top_k)

    return {
        "query": query,
        "before_rerank": before,
        "after_rerank": after
    }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.vector_store import load_index
    from src.bm25_retriever import load_bm25_index
    from src.hybrid_retriever import hybrid_search
    from src.embedder import load_embedding_model

    # Load everything
    print("Loading indexes...")
    faiss_index, chunks = load_index()
    bm25_index = load_bm25_index()
    model = load_embedding_model()
    reranker = load_reranker()

    # Test query
    query = "How does multi-head attention work in transformers?"

    print(f"\nQuery: '{query}'")

    # Step 1: Hybrid search gets top 10
    print("\n[1] Running hybrid search (top 10)...")
    hybrid_results = hybrid_search(
        query, faiss_index, bm25_index, chunks, model, top_k=10
    )

    # Step 2: Rerank to get top 5
    print("\n[2] Reranking to get top 5...")
    reranked = rerank_chunks(query, hybrid_results, top_k=5)

    # Show before reranking
    print("\n--- BEFORE RERANKING (top 5 of hybrid) ---")
    for i, r in enumerate(hybrid_results[:5]):
        print(f"  [{i+1}] {r['source']} | Page {r['page']}")
        print(f"       RRF Score: {r.get('rrf_score', 0):.4f}")
        print(f"       {r['text'][:150]}...")
        print()

    # Show after reranking
    print("\n--- AFTER RERANKING (top 5 by cross-encoder) ---")
    for i, r in enumerate(reranked):
        print(f"  [{i+1}] {r['source']} | Page {r['page']}")
        print(f"       Rerank Score: {r['rerank_score']:.4f}")
        print(f"       {r['text'][:150]}...")
        print()
