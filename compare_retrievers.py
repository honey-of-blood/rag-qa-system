from src.vector_store import load_index, search_faiss
from src.bm25_retriever import load_bm25_index, search_bm25
from src.hybrid_retriever import hybrid_search
from src.embedder import load_embedding_model


def compare_all_retrievers(query: str):
    """
    Runs the same query through all 3 retrievers
    and prints results side by side for comparison.
    """
    faiss_index, chunks = load_index()
    bm25_index = load_bm25_index()
    model = load_embedding_model()

    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print(f"{'='*60}")

    # Semantic only
    print("\n--- SEMANTIC ONLY (FAISS) ---")
    query_embedding = model.encode([query], convert_to_numpy=True)
    semantic = search_faiss(query_embedding, faiss_index, chunks, top_k=3)
    for i, r in enumerate(semantic):
        print(f"  [{i+1}] {r['source']} | Page {r['page']}")
        print(f"       {r['text'][:150]}...")

    # Keyword only
    print("\n--- KEYWORD ONLY (BM25) ---")
    keyword = search_bm25(query, bm25_index, chunks, top_k=3)
    for i, r in enumerate(keyword):
        print(f"  [{i+1}] {r['source']} | Page {r['page']}")
        print(f"       {r['text'][:150]}...")

    # Hybrid
    print("\n--- HYBRID (FAISS + BM25 + RRF) ---")
    hybrid = hybrid_search(
        query, faiss_index, bm25_index, chunks, model, top_k=3
    )
    for i, r in enumerate(hybrid):
        print(f"  [{i+1}] {r['source']} | Page {r['page']}")
        print(f"       RRF Score: {r['rrf_score']:.4f}")
        print(f"       {r['text'][:150]}...")


if __name__ == "__main__":
    queries = [
        "multi head attention mechanism",
        "gradient descent optimization",
        "what is a transformer architecture"
    ]
    for q in queries:
        compare_all_retrievers(q)
