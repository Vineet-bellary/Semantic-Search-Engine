from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"

INGESTED_DATA_DIR = ROOT_DIR / "ingested_data"

# Paramters
CHUNK_SIZE = 175
OVERLAP_SIZE = 0

CONFIDENCE_THRESHOLD = 0.1
