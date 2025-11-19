"""Script to load embeddings from FAISS index files with logging."""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, get_logger
import os

logger = get_logger(__name__)


def load_faiss_index():
    """
    Load the FAISS index saved locally. Uses the same HuggingFace embedding
    wrapper that you used to create the index so that vector shapes match.
    """
    logger.info("Loading FAISS index from: %s", FAISS_INDEX_PATH)
    
    # Determine the folder path and index name
    index_path = FAISS_INDEX_PATH
    
    # If the path ends with .faiss, it's a file path - extract folder and name
    if index_path.endswith('.faiss'):
        folder_path = os.path.dirname(index_path)
        index_name = os.path.splitext(os.path.basename(index_path))[0]
        logger.debug("Parsed as file path - folder: %s, name: %s", folder_path, index_name)
    # If it's a folder path
    elif os.path.isdir(index_path):
        folder_path = index_path
        index_name = None
        logger.debug("Using folder path: %s", folder_path)
    # If it's a file that exists (without .faiss extension)
    elif os.path.isfile(index_path):
        folder_path = os.path.dirname(index_path)
        index_name = os.path.splitext(os.path.basename(index_path))[0]
        logger.debug("Parsed existing file - folder: %s, name: %s", folder_path, index_name)
    # Default: assume it's a folder path and let validation catch missing files
    else:
        folder_path = index_path
        index_name = None
        logger.debug("Treating as folder path (not yet validated): %s", folder_path)

    # Validate that the folder exists
    if not os.path.exists(folder_path):
        logger.error("FAISS folder not found: %s", folder_path)
        logger.error("Current working directory: %s", os.getcwd())
        logger.error("FAISS_INDEX_PATH was: %s", FAISS_INDEX_PATH)
        raise FileNotFoundError(
            f"FAISS folder not found: {folder_path}. "
            f"Please ensure the folder exists at: {os.path.abspath(folder_path)}"
        )

    # If an explicit index name is known, ensure the file exists
    if index_name is not None:
        expected_file = os.path.join(folder_path, f"{index_name}.faiss")
        if not os.path.exists(expected_file):
            logger.error("FAISS index file not found: %s", expected_file)
            logger.error("Available files in %s:", folder_path)
            try:
                files = os.listdir(folder_path)
                for f in files:
                    logger.error("  - %s", f)
            except Exception as e:
                logger.error("Could not list folder contents: %s", e)
            raise FileNotFoundError(
                f"FAISS index file not found: {expected_file}. "
                f"Please ensure the index exists at: {os.path.abspath(expected_file)}"
            )

    # Load the embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("HuggingFace embeddings model loaded: %s", EMBEDDING_MODEL)
    except Exception as e:
        logger.exception("Failed to load HuggingFace embeddings: %s", e)
        raise

    # Prepare load kwargs
    load_kwargs = {
        "folder_path": folder_path,
        "embeddings": embeddings,
        "allow_dangerous_deserialization": True,
    }

    if index_name is not None:
        load_kwargs["index_name"] = index_name
        logger.debug("Loading with index_name: %s", index_name)

    try:
        db = FAISS.load_local(**load_kwargs)
        logger.info("FAISS Index Successfully Loaded from: %s", folder_path)
        return db
    except Exception as e:
        logger.exception("Failed to load FAISS index: %s", e)
        raise