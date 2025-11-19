# Configuration Settings for the Backend Application
import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
# Look for .env in multiple locations
env_paths = [
    os.path.join(os.path.dirname(__file__), '.env'),  # backend/.env
    os.path.join(os.path.dirname(__file__), '..', '.env'),  # project root/.env
]

loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        loaded = True
        print(f"✅ Loaded .env from: {env_path}")
        break

if not loaded:
    print(f"⚠️  No .env file found. Checked locations:")
    for path in env_paths:
        print(f"   - {path}")

# Verify API key is loaded
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in environment!")
    print("   Please create a .env file with: GEMINI_API_KEY=your_key_here")
    print(f"   Place it in: {os.path.dirname(__file__)}")
    sys.exit(1)
else:
    print(f"✅ API Key loaded (length: {len(GEMINI_API_KEY)} characters)")

# Set the API key for Google GenAI
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY  # ADK uses GOOGLE_API_KEY

# Defining project root directory - go up from backend/ folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# FAISS index path - use folder path, not the .faiss file directly
# This should point to the folder containing the index files
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")

# Create the folder if it doesn't exist (for clarity)
# Note: This doesn't create the actual FAISS index - that should be done separately
if not os.path.exists(FAISS_INDEX_PATH):
    print(f"⚠️  FAISS folder does not exist: {FAISS_INDEX_PATH}")
    print(f"   Please ensure your FAISS index files are in this location")
    print(f"   Expected files: index.faiss, index.pkl")

print(f"📁 FAISS Index Path: {FAISS_INDEX_PATH}")

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Google Model Settings
ADK_MODEL_NAME = "gemini-2.5-flash"

# Retrieval Settings
TOP_K = 3


# Logging configuration: configure root logger for the application
LOG_LEVEL = os.environ.get("FNA_LOG_LEVEL", "INFO").upper()

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
root_logger = logging.getLogger()
if not root_logger.handlers:
    root_logger.addHandler(_handler)
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

def get_logger(name: str):
    """Helper to get a logger configured with the app's root settings."""
    return logging.getLogger(name)