# backend/config.py
"""
Configuration for ADK-based Fact-Checking Agent
"""
import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
env_paths = [
    os.path.join(os.path.dirname(__file__), '.env'),
    os.path.join(os.path.dirname(__file__), '..', '.env'),
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found!")
    sys.exit(1)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")

# Model Configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
ADK_MODEL_NAME = "gemini-2.5-flash"  # ADK-compatible model

# Retrieval Configuration
FAISS_TOP_K = 5
GOOGLE_TOP_K = 10
FINAL_TOP_K = 10  # After ranking

# Logging
LOG_LEVEL = os.environ.get("FNA_LOG_LEVEL", "WARNING").upper()

_formatter = logging.Formatter("%(levelname)s: %(message)s")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_formatter)

root_logger = logging.getLogger()
if not root_logger.handlers:
    root_logger.addHandler(_handler)
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

# Suppress verbose libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

def get_logger(name: str):
    """Get configured logger"""
    return logging.getLogger(name)