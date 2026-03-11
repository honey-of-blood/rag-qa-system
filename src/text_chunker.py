from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(pages: list[dict], chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """
    Takes a list of page dicts and splits them into smaller chunks.
    Preserves metadata: source filename and page number.
    Returns a list of chunk dicts with keys: text, source, page, chunk_id
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_id = 0

    for page in pages:
        splits = splitter.split_text(page["text"])

        for split in splits:
            if split.strip():  # skip empty chunks
                all_chunks.append({
                    "text": split.strip(),
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": chunk_id
                })
                chunk_id += 1

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    from document_loader import load_pdfs_from_folder

    # Make path independent of where script is run
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data"

    pages = load_pdfs_from_folder(str(DATA_PATH))

    if not pages:
        print("No pages loaded. Check if PDFs exist in the data folder.")
        exit()

    chunks = chunk_documents(pages)

    if not chunks:
        print("No chunks created.")
        exit()

    print("\n--- Sample Chunk ---")
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Source: {chunks[0]['source']}")
    print(f"Page: {chunks[0]['page']}")
    print(f"Text preview: {chunks[0]['text'][:300]}")
    print(f"\nChunk length (chars): {len(chunks[0]['text'])}")

    print("\n--- Chunk Size Distribution ---")

    lengths = [len(c["text"]) for c in chunks]

    print(f"Min: {min(lengths)}")
    print(f"Max: {max(lengths)}")
    print(f"Average: {sum(lengths) // len(lengths)}")
