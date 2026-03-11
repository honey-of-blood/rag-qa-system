from src.vector_store import load_index
from src.bm25_retriever import load_bm25_index
from src.hybrid_retriever import hybrid_search
from src.reranker import load_reranker, rerank_chunks
from src.embedder import load_embedding_model


def run_phase5_test():
    print("=" * 50)
    print("PHASE 5 TEST: Cross-Encoder Reranking")
    print("=" * 50)

    # Load everything
    print("\n[1] Loading all indexes and models...")
    faiss_index, chunks = load_index()
    bm25_index = load_bm25_index()
    model = load_embedding_model()
    reranker = load_reranker()

    # Test queries
    test_queries = [
        "How does scaled dot product attention work?",
        "What is the difference between encoder and decoder?",
        "Explain low rank matrix decomposition for fine tuning"
    ]

    print("\n[2] Running full pipeline: hybrid search → rerank\n")

    for query in test_queries:
        print(f"{'='*50}")
        print(f"Query: '{query}'")

        # Hybrid search
        hybrid_results = hybrid_search(
            query, faiss_index, bm25_index, chunks, model, top_k=10
        )
        print(f"Hybrid search returned: {len(hybrid_results)} chunks")

        # Rerank
        reranked = rerank_chunks(query, hybrid_results, top_k=5)
        print(f"After reranking, top 5 results:")

        for i, r in enumerate(reranked):
            print(f"\n  [{i+1}] {r['source']} | Page {r['page']}")
            print(f"       Rerank Score: {r['rerank_score']:.4f}")
            print(f"       Preview: {r['text'][:200]}...")

        print()

    print("=" * 50)
    print("PHASE 5 COMPLETE")
    print("Cross-encoder reranking pipeline working end to end")
    print("=" * 50)


if __name__ == "__main__":
    run_phase5_test()
