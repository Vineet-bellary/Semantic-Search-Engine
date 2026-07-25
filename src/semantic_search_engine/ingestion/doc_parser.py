from pathlib import Path

def parse_doc(pdf_path: Path, converter):
    if converter is None:
        raise ValueError(
            "DocumentConverter instance is required for parsing documents."
        )

    doc_object = converter.convert(pdf_path)

    return doc_object.document
