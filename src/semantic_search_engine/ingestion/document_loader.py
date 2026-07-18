from pathlib import Path

data_dir = Path(__file__).parents[3] / "data"


def normalize_filename(filename: str) -> str:
    """
    Normalize the filename by removing spaces and converting to lowercase.
    """
    return filename.replace(" ", "_").lower()


def get_file_path(data_dir: Path) -> list[Path]:
    file_paths = []
    for file in data_dir.iterdir():
        if file.suffix.lower() == ".pdf":
            file_paths.append(file.resolve())
    return file_paths

