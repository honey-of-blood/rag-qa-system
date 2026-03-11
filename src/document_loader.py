import os
from pathlib import Path
from pypdf import PdfReader


def load_pdfs_from_folder(folder_path: str) -> list[dict]:
    """
    Loads all PDFs from a folder.
    Returns a list of dicts with keys: text, source, page
    """
    folder = Path(folder_path)
    all_pages = []

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():  # skip empty pages
                    all_pages.append({
                        "text": text.strip(),
                        "source": pdf_path.name,
                        "page": page_num + 1
                    })
        except Exception as e:
            print(f"Error loading {pdf_path.name}: {e}")

    print(f"Total pages loaded: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    pages = load_pdfs_from_folder("data/")
    print("\n--- Sample Output ---")
    print(f"Source: {pages[0]['source']}")
    print(f"Page: {pages[0]['page']}")
    print(f"Text preview: {pages[0]['text'][:300]}")
