from src.bm25_retriever import search_bm25
from src.vector_store import search_faiss
from sentence_transformers import SentenceTransformer


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Combines semantic and keyword results using
    Reciprocal Rank Fusion (RRF) scoring.
    Higher score = more relevant.
    RRF formula: score = 1 / (k + rank)
    """
    scores = {}

    # Score semantic results
    for rank, chunk in enumerate(semantic_results):
        chunk_id = chunk["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    # Score keyword results
    for rank, chunk in enumerate(keyword_results):
        chunk_id = chunk["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    # Build a lookup dict by chunk_id
    all_chunks = {c["chunk_id"]: c for c in semantic_results + keyword_results}

    # Sort by combined RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results = []
    for cid in sorted_ids:
        chunk = all_chunks[cid].copy()
        chunk["rrf_score"] = scores[cid]
        chunk["retrieval_type"] = "hybrid"
        fused_results.append(chunk)

    return fused_results


def hybrid_search(
    query: str,
    faiss_index,
    bm25_index,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 10
) -> list[dict]:
    """
    Full hybrid search pipeline:
    1. Semantic search via FAISS
    2. Keyword search via BM25
    3. Combine with Reciprocal Rank Fusion
    4. Return top_k deduplicated results
    """
    # Semantic retrieval
    query_embedding = model.encode([query], convert_to_numpy=True)
    semantic_results = search_faiss(
        query_embedding, faiss_index, chunks, top_k=top_k
    )

    # Keyword retrieval
    keyword_results = search_bm25(
        query, bm25_index, chunks, top_k=top_k
    )

    # Fuse results
    fused = reciprocal_rank_fusion(semantic_results, keyword_results)

    # Return top_k
    return fused[:top_k]


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.vector_store import load_index
    from src.bm25_retriever import load_bm25_index, build_bm25_index, save_bm25_index
    from src.embedder import load_embedding_model

    # Load everything
    faiss_index, chunks = load_index()
    model = load_embedding_model()

    try:
        bm25_index = load_bm25_index()
    except FileNotFoundError:
        bm25_index = build_bm25_index(chunks)
        save_bm25_index(bm25_index)

    # Test queries
    test_queries = [
        "How does self attention work in transformers?",
        "What is retrieval augmented generation?",
        "Explain fine tuning with low rank adaptation"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{query}'")
        print(f"{'='*50}")

        results = hybrid_search(
            query, faiss_index, bm25_index, chunks, model, top_k=5
        )

        for i, r in enumerate(results):
            print(f"\n  [{i+1}] Source: {r['source']} | Page: {r['page']}")
            print(f"       RRF Score: {r['rrf_score']:.4f}")
            print(f"       Preview: {r['text'][:200]}...")
