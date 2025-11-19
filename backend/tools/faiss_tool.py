"""Wrapping the FAISS search functionality with Google ADK Tools and logging."""
from google.adk.tools import LongRunningFunctionTool
from loader.embeddings_loader import load_faiss_index
from config import TOP_K, get_logger

logger = get_logger(__name__)

# Singleton pattern to avoid reloading the FAISS index multiple times
_db = None
def _get_db():
    global _db
    if _db is None:
        logger.info("Loading FAISS DB for tool usage")
        _db = load_faiss_index()
    return _db

# Defining the FAISS search tool
def faiss_search(query: str, k: int = TOP_K):
    """
    Return top-k documents from the FAISS index as plain dicts
    with 'content' and 'meta' fields so the agent can consume them.
    """
    logger.info("faiss_search called query=%s k=%d", query, k)
    db = _get_db()
    docs = db.similarity_search(query, k=k)

    results = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        results.append({
            "content": d.page_content,
            "meta": meta
        })
    logger.debug("faiss_search returning %d results", len(results))
    return results

# Wrapping the faiss_search function as a FunctionTool for ADK
# Wrap as ADK FunctionTool
faiss_search_tool = LongRunningFunctionTool(
    func=faiss_search
)