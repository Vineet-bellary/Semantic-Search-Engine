from pathlib import Path
import re

from semantic_search_engine.ingestion.document_loader import get_file_path


def load_markdown(md_path: Path) -> list[str]:
    """
    Load a Markdown file and return its content as a list of lines.

    Args:
        md_path (Path): Path to the Markdown file.

    Returns:
        list[str]: List of lines in the Markdown file.
    """

    with open(md_path, "r", encoding="utf-8") as f:
        return f.readlines()


def get_heading(line: str):
    """
    Get the heading level and title from a Markdown line.

    Args:
        line (str): A line from a Markdown file.

    Returns:
        tuple[int, str] | None: A tuple containing the heading level and title, or None if the line is not a heading.
    """
    line = line.strip()

    if not line:
        return None

    # Markdown headings
    if line.startswith("#"):
        level = len(line.split(" ")[0])
        title = line.replace("#", "").strip()

        return level, title

    # Chapter headings
    chapter_pattern = r"^Chapter\s+\d+[:\s].+"

    if re.match(chapter_pattern, line, re.IGNORECASE):
        return 1, line

    # Numbered headings
    numbered_pattern = r"^\d+(\.\d+)*\.?\s+[A-Z].+"

    if re.match(numbered_pattern, line):
        return 2, line

    return None


def extract_sections_from_markdown(md_path: Path) -> list[dict[str, str]]:
    """
    Extract sections from a Markdown file based on headings.

    Args:
        md_path (Path): Path to the Markdown file.

    Returns:
        list[dict[str, str]]: List of sections extracted from the Markdown file.
    Each section is represented as a dictionary with the following keys:
        - "heading_path": List of headings leading to the section.
        - "content": The content of the section.
    """
    lines = load_markdown(md_path)

    sections = []

    heading_path = []
    current_content = []

    for line in lines:

        heading = get_heading(line)

        if heading:

            # save previous section
            if current_content and "".join(current_content).strip():
                sections.append(
                    {
                        "heading_path": heading_path.copy(),
                        "content": "\n".join(current_content).strip(),
                    }
                )

                current_content = []

            level, title = heading

            # update hierarchy
            heading_path = heading_path[: level - 1]
            heading_path.append(title)

        else:
            current_content.append(line)

    # save final section
    if current_content and "".join(current_content).strip():
        sections.append(
            {
                "heading_path": heading_path.copy(),
                "content": "\n".join(current_content).strip(),
            }
        )

    return sections