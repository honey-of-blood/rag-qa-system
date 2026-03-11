from sentence_transformers import SentenceTransformer
import numpy as np

# Using a free, lightweight model that runs on CPU
MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model():
    """
    Loads the sentence transformer model.
    Downloads automatically on first run (~90MB).
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Embedding model loaded successfully")
    return model


def generate_embeddings(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Takes a list of chunk dicts and generates embeddings for each.
    Returns a numpy array of shape (num_chunks, embedding_dim)
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")
    print("This may take a few minutes on first run...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    from document_loader import load_pdfs_from_folder
    from text_chunker import chunk_documents

    pages = load_pdfs_from_folder("../data/")
    chunks = chunk_documents(pages)

    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)

    print("\n--- Sample Embedding ---")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First 5 values of chunk 0: {embeddings[0][:5]}")
