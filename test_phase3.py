from src.document_loader import load_pdfs_from_folder
from src.text_chunker import chunk_documents
from src.embedder import load_embedding_model, generate_embeddings
from src.vector_store import (
    build_faiss_index,
    save_index,
    load_index,
    search_faiss
)


def run_phase3_test():
    print("=" * 50)
    print("PHASE 3 TEST: Embedding + Vector Store")
    print("=" * 50)

    # Step 1: Load and chunk
    print("\n[1] Loading and chunking PDFs...")
    pages = load_pdfs_from_folder("data/")
    chunks = chunk_documents(pages)

    # Step 2: Load embedding model
    print("\n[2] Loading embedding model...")
    model = load_embedding_model()

    # Step 3: Generate embeddings
    print("\n[3] Generating embeddings...")
    embeddings = generate_embeddings(chunks, model)

    # Step 4: Build and save FAISS index
    print("\n[4] Building and saving FAISS index...")
    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    # Step 5: Reload from disk to verify persistence
    print("\n[5] Reloading index from disk...")
    index, chunks = load_index()

    # Step 6: Run test queries
    test_queries = [
        "What is the attention mechanism?",
        "How does fine tuning work?",
        "What is retrieval augmented generation?"
    ]

    print("\n[6] Running test queries...\n")
    for query in test_queries:
        print(f"Query: '{query}'")
        query_embedding = model.encode([query], convert_to_numpy=True)
        results = search_faiss(query_embedding, index, chunks, top_k=3)

        for i, r in enumerate(results):
            print(f"  [{i+1}] {r['source']} | Page {r['page']} | Score: {r['score']:.4f}")
            print(f"       {r['text'][:150]}...")
        print()

    print("=" * 50)
    print("PHASE 3 COMPLETE")
    print(f"Total chunks embedded: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index saved to disk: models/faiss_index.bin")
    print("=" * 50)


if __name__ == "__main__":
    run_phase3_test()
