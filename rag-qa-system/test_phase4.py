from src.vector_store import load_index
from src.bm25_retriever import build_bm25_index, save_bm25_index, load_bm25_index
from src.hybrid_retriever import hybrid_search
from src.embedder import load_embedding_model


def run_phase4_test():
    print("=" * 50)
    print("PHASE 4 TEST: BM25 + Hybrid Retrieval")
    print("=" * 50)

    # Load FAISS index and chunks
    print("\n[1] Loading FAISS index and chunks...")
    faiss_index, chunks = load_index()

    # Build BM25
    print("\n[2] Building BM25 index...")
    bm25_index = build_bm25_index(chunks)
    save_bm25_index(bm25_index)

    # Reload BM25 from disk
    print("\n[3] Reloading BM25 from disk...")
    bm25_index = load_bm25_index()

    # Load embedding model
    print("\n[4] Loading embedding model...")
    model = load_embedding_model()

    # Run hybrid search
    print("\n[5] Running hybrid search test queries...\n")
    test_queries = [
        "explain the attention mechanism",
        "what is low rank adaptation for fine tuning",
        "retrieval augmented generation pipeline"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        results = hybrid_search(
            query, faiss_index, bm25_index, chunks, model, top_k=3
        )
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r['source']} | Page {r['page']} | RRF: {r['rrf_score']:.4f}")
        print()

    print("=" * 50)
    print("PHASE 4 COMPLETE")
    print(f"BM25 index saved: models/bm25_index.pkl")
    print(f"Hybrid search working with RRF fusion")
    print("=" * 50)


if __name__ == "__main__":
    run_phase4_test()
