"""Wrapping the FAISS search functionality with Google ADK Tools."""
from google.adk.tools import FunctionTool
from loader.embeddings_loader import load_faiss_index
from config import TOP_K, FAISS_INDEX_PATH, get_logger
import os

logger = get_logger(__name__)

# Singleton pattern to avoid reloading the FAISS index multiple times
_db = None
_load_error = None

def _get_db():
    """Lazy load the FAISS database."""
    global _db, _load_error
    
    if _load_error is not None:
        # Already tried and failed, return the cached error
        raise _load_error
    
    if _db is None:
        try:
            logger.info("Loading FAISS DB for tool usage")
            logger.info("Expected FAISS path: %s", FAISS_INDEX_PATH)
            
            # Check if the path exists
            if not os.path.exists(FAISS_INDEX_PATH):
                error_msg = (
                    f"FAISS index path does not exist: {FAISS_INDEX_PATH}\n"
                    f"Absolute path: {os.path.abspath(FAISS_INDEX_PATH)}\n"
                    f"Current working directory: {os.getcwd()}\n"
                    f"Please ensure you have created the FAISS index and placed it in the correct location."
                )
                logger.error(error_msg)
                _load_error = FileNotFoundError(error_msg)
                raise _load_error
            
            _db = load_faiss_index()
            logger.info("FAISS DB loaded successfully")
        except Exception as e:
            _load_error = e
            logger.exception("Failed to load FAISS DB: %s", e)
            raise
    
    return _db


def faiss_search(query: str, k: int = TOP_K) -> list:
    """
    Search the FAISS knowledge base for relevant documents.
    
    Args:
        query: The search query to find relevant documents
        k: Number of top results to return (default: 5)
    
    Returns:
        A list of relevant document excerpts with their metadata,
        or a list with an error message if the index is not available
    """
    logger.info("faiss_search called query='%s' k=%d", query, k)
    
    try:
        db = _get_db()
        docs = db.similarity_search(query, k=k)
        
        results = []
        for i, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": meta.get("source", "unknown"),
                "metadata": meta
            })
        
        logger.debug("faiss_search returning %d results", len(results))
        return results
        
    except FileNotFoundError as e:
        error_msg = (
            f"FAISS index not found. Please ensure the FAISS index has been created. "
            f"Error: {str(e)}"
        )
        logger.error(error_msg)
        return [{"error": error_msg, "type": "not_found"}]
    except Exception as e:
        error_msg = f"FAISS search failed: {str(e)}"
        logger.exception(error_msg)
        return [{"error": error_msg, "type": "search_error"}]


# Create the ADK FunctionTool
faiss_search_tool = FunctionTool(func=faiss_search)