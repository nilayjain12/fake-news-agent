"""Script to load embeddings from FAISS index files with logging."""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, get_logger
import os

logger = get_logger(__name__)


# Function to load embeddings from FAISS index
def load_faiss_index():
    """
    Load the FAISS index saved locally. Uses the same HuggingFace embedding
    wrapper that you used to create the index so that vector shapes match.
    """
    logger.info("Loading FAISS index from: %s", FAISS_INDEX_PATH)

    # Accept either a folder path or a full path to the index file.
    if os.path.isfile(FAISS_INDEX_PATH):
        folder_path = os.path.dirname(FAISS_INDEX_PATH)
        index_name = os.path.splitext(os.path.basename(FAISS_INDEX_PATH))[0]
    else:
        folder_path = FAISS_INDEX_PATH
        index_name = None

    if not os.path.exists(folder_path):
        logger.error("FAISS folder not found: %s", folder_path)
        raise FileNotFoundError(
            f"FAISS folder not found: {folder_path}. Ensure the index files exist."
        )

    # If an explicit index name is known, ensure the file exists
    if index_name is not None:
        expected_file = os.path.join(folder_path, f"{index_name}.faiss")
        if not os.path.exists(expected_file):
            logger.error("FAISS index file not found: %s", expected_file)
            raise FileNotFoundError(
                f"FAISS index file not found: {expected_file}."
            )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    load_kwargs = {
        "folder_path": folder_path,
        "embeddings": embeddings,
        "allow_dangerous_deserialization": True,
    }

    if index_name is not None:
        load_kwargs["index_name"] = index_name

    db = FAISS.load_local(**load_kwargs)
    logger.info("---FAISS Index Loaded---")
    return db