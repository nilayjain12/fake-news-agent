# backend/config.py
# Configuration Settings for the Backend Application
import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
env_paths = [
    os.path.join(os.path.dirname(__file__), '.env'),
    os.path.join(os.path.dirname(__file__), '..', '.env'),
]

loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        loaded = True
        break

# Verify API key is loaded
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in environment!")
    sys.exit(1)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
ADK_MODEL_NAME = "gemini-2.5-flash"
TOP_K = 3

# ============ LOGGING CONFIGURATION ============
# Set to WARNING to suppress debug/info logs, INFO for minimal output
LOG_LEVEL = os.environ.get("FNA_LOG_LEVEL", "WARNING").upper()

# Custom formatter: minimal output
_formatter = logging.Formatter("%(levelname)s: %(message)s")

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_formatter)

root_logger = logging.getLogger()
if not root_logger.handlers:
    root_logger.addHandler(_handler)
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

# Suppress verbose third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

def get_logger(name: str):
    """Get a logger configured with minimal output."""
    return logging.getLogger(name)