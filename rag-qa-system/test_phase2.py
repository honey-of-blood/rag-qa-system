from src.document_loader import load_pdfs_from_folder
from src.text_chunker import chunk_documents
import json


def run_phase2_test():
    print("=" * 50)
    print("PHASE 2 TEST: PDF Ingestion & Chunking")
    print("=" * 50)

    # Step 1: Load PDFs
    print("\n[1] Loading PDFs...")
    pages = load_pdfs_from_folder("data/")

    if not pages:
        print("ERROR: No pages loaded. Check your data/ folder.")
        return

    # Step 2: Chunk documents
    print("\n[2] Chunking documents...")
    chunks = chunk_documents(pages)

    if not chunks:
        print("ERROR: No chunks created.")
        return

    # Step 3: Show sample results
    print("\n[3] Sample chunks from different sources:")
    seen_sources = set()
    for chunk in chunks:
        if chunk["source"] not in seen_sources:
            print(f"\n--- {chunk['source']} ---")
            print(f"  Page: {chunk['page']}")
            print(f"  Chunk ID: {chunk['chunk_id']}")
            print(f"  Preview: {chunk['text'][:200]}...")
            seen_sources.add(chunk["source"])

    # Step 4: Save sample to file for inspection
    sample = chunks[:10]
    with open("sample_chunks.json", "w") as f:
        json.dump(sample, f, indent=2)
    print("\n[4] First 10 chunks saved to sample_chunks.json for inspection")

    print("\n" + "=" * 50)
    print("PHASE 2 COMPLETE")
    print(f"Total PDFs processed: {len(set(c['source'] for c in chunks))}")
    print(f"Total pages loaded: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")
    print("=" * 50)


if __name__ == "__main__":
    run_phase2_test()
