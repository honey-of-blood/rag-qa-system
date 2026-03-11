from src.vector_store import load_index
from src.bm25_retriever import load_bm25_index
from src.hybrid_retriever import hybrid_search
from src.reranker import rerank_chunks
from src.embedder import load_embedding_model


# Define test questions with known expected keywords
# These keywords should appear in a truly relevant chunk
TEST_CASES = [
    {
        "query": "What is multi-head attention?",
        "expected_keywords": ["attention", "head", "queries", "keys", "values"]
    },
    {
        "query": "How does positional encoding work?",
        "expected_keywords": ["positional", "encoding", "position", "sequence"]
    },
    {
        "query": "What is the purpose of layer normalization?",
        "expected_keywords": ["normalization", "layer", "mean", "variance"]
    },
    {
        "query": "Explain the encoder decoder architecture",
        "expected_keywords": ["encoder", "decoder", "stack", "layers"]
    },
    {
        "query": "What is retrieval augmented generation?",
        "expected_keywords": ["retrieval", "generation", "document", "knowledge"]
    }
]


def keyword_relevance_score(chunk_text: str, keywords: list[str]) -> float:
    """
    Simple relevance check: what fraction of expected
    keywords appear in the chunk text?
    """
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)


def evaluate(faiss_index, bm25_index, chunks, model):
    print("=" * 60)
    print("RERANKER EVALUATION")
    print("=" * 60)

    before_scores = []
    after_scores = []

    for tc in TEST_CASES:
        query = tc["query"]
        keywords = tc["expected_keywords"]

        # Get hybrid results
        hybrid_results = hybrid_search(
            query, faiss_index, bm25_index, chunks, model, top_k=10
        )

        if not hybrid_results:
            continue

        # Score top result BEFORE reranking
        top_before = hybrid_results[0]
        score_before = keyword_relevance_score(top_before["text"], keywords)

        # Rerank and score top result AFTER reranking
        reranked = rerank_chunks(query, hybrid_results, top_k=5)
        top_after = reranked[0]
        score_after = keyword_relevance_score(top_after["text"], keywords)

        before_scores.append(score_before)
        after_scores.append(score_after)

        print(f"\nQuery: '{query}'")
        print(f"  Before rerank top result: {top_before['source']} | Page {top_before['page']}")
        print(f"  Keyword relevance score:  {score_before:.2f}")
        print(f"  After rerank top result:  {top_after['source']} | Page {top_after['page']}")
        print(f"  Keyword relevance score:  {score_after:.2f}")
        improvement = ((score_after - score_before) / max(score_before, 0.01)) * 100
        print(f"  Improvement: {improvement:+.1f}%")

    avg_before = sum(before_scores) / len(before_scores)
    avg_after = sum(after_scores) / len(after_scores)
    overall_improvement = ((avg_after - avg_before) / max(avg_before, 0.01)) * 100

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Average relevance BEFORE reranking: {avg_before:.2f}")
    print(f"Average relevance AFTER reranking:  {avg_after:.2f}")
    print(f"Overall improvement: {overall_improvement:+.1f}%")
    print(f"{'='*60}")

    return overall_improvement


if __name__ == "__main__":
    print("Loading all indexes...")
    faiss_index, chunks = load_index()
    bm25_index = load_bm25_index()
    model = load_embedding_model()

    evaluate(faiss_index, bm25_index, chunks, model)
