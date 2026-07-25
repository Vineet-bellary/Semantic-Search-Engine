from pathlib import Path
# from docling.document_converter import DocumentConverter

from semantic_search_engine.ingestion.document_loader import get_file_path


def pdf_to_markdown(pdf_path: Path, output_dir: Path, converter) -> Path:
    """
    Convert a PDF document to Markdown format.

    Args:
        pdf_path (Path): Path to the input PDF file.
        output_dir (Path): Directory where the output Markdown file will be saved.

    Raises:
        ValueError: If the PDF file cannot be converted to Markdown.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting now...")
    doc_object = converter.convert(pdf_path)
    md_text = doc_object.document.export_to_markdown()
    print(f"Converted documents to markdown: {pdf_path}")

    # Save the converted document as a Markdown file
    output_path = Path(output_dir) / f"{Path(pdf_path).stem}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return output_path

