# backend/tools/faiss_tool.py
"""FAISS search functionality as a Google ADK Tool."""
from google.adk.tools import FunctionTool
from loader.embeddings_loader import load_faiss_index
from config import TOP_K, FAISS_INDEX_PATH, get_logger
import os

logger = get_logger(__name__)

_db = None
_load_error = None

def _get_db():
    """Lazy load the FAISS database."""
    global _db, _load_error
    
    if _load_error is not None:
        raise _load_error
    
    if _db is None:
        try:
            _db = load_faiss_index()
        except Exception as e:
            _load_error = e
            raise
    
    return _db


def faiss_search(query: str, k: int = TOP_K) -> list:
    """Search the FAISS knowledge base for relevant documents."""
    logger.warning("🔎 FAISS search: %s", query[:60])
    
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
        
        logger.warning("   → Found %d results", len(results))
        return results
        
    except FileNotFoundError as e:
        logger.warning("❌ FAISS index not found")
        return [{"error": f"FAISS index not found: {str(e)}", "type": "not_found"}]
    except Exception as e:
        logger.warning("❌ FAISS search error: %s", str(e)[:50])
        return [{"error": f"FAISS search failed: {str(e)}", "type": "search_error"}]


faiss_search_tool = FunctionTool(func=faiss_search)