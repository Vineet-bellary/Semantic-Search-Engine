from pathlib import Path
from pypdf import PdfReader

# print("pypdf version:", pypdf.__version__)
data_dir = Path(__file__).parent.parent.parent.parent / "data"
pdf_path = data_dir / "semantic_search_test_corpus.pdf"


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    if pdf_path.exists():
        print(f"PDF file exists: {pdf_path}")
    else:
        print(f"PDF file does not exist: {pdf_path}")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total number of pages in the PDF: {total_pages}\n")

    pdf_page_data = []
    for page_number, page in enumerate(reader.pages):
        pdf_page_data.append(
            {
                "document_name": pdf_path.name,
                "page_number": page_number + 1,
                "text": page.extract_text() or "",
            }
        )
    return pdf_page_data
