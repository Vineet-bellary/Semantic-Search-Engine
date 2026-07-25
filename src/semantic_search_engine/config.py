from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[2]
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_ID")

# Storage directories
DATA_DIR = ROOT_DIR / "data"
TEST_DIR = ROOT_DIR / "test"
INGESTED_DATA_DIR = ROOT_DIR / "ingested_data"

# Paramters
CONFIDENCE_THRESHOLD = 0.1

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
