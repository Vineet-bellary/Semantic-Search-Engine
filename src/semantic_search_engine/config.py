from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[2]
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_ID")

# Storage directories
DATA_DIR = ROOT_DIR / "data"
INGESTED_DATA_DIR = ROOT_DIR / "ingested_data"

# Paramters
CHUNK_SIZE = 250
OVERLAP_SIZE = 0

CONFIDENCE_THRESHOLD = 0.6

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
