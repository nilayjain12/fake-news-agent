# Configuration Settings for the Backend Application
import os
import logging
import sys

# Defining project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# FAISS index path
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "index.faiss")

# Embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Google Model Settings
ADK_MODEL_NAME = "gemini-2.5-flash"

# Retrieval Settings
TOP_K = 5


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