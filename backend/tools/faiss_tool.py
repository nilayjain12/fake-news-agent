# backend/tools/faiss_tool.py
"""FAISS search functionality as a Google ADK Tool."""
from google.adk.tools import FunctionTool
from loader.embeddings_loader import load_faiss_index
from config import TOP_K, FAISS_INDEX_PATH, get_logger
import os
import datetime

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
    """Search FAISS with date-based relevance weighting."""
    logger.warning("🔎 FAISS search: %s", query[:60])
    
    try:
        db = _get_db()
        docs = db.similarity_search(query, k=k)
        
        results = []
        current_time = datetime.datetime.now()
        
        for i, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            
            # Try to extract date from metadata
            doc_date = meta.get("date")
            age_days = None
            
            if doc_date:
                try:
                    doc_datetime = datetime.fromisoformat(str(doc_date))
                    age_days = (current_time - doc_datetime).days
                except:
                    pass
            
            # Mark old documents
            is_old = age_days > 90 if age_days else False
            
            results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
                "age_days": age_days,
                "is_old_source": is_old,  # FLAG for old data
                "freshness_warning": "⚠️ This source is from an older date" if is_old else ""
            })
        
        logger.warning("   → Found %d results (with freshness flags)", len(results))
        return results
        
    except Exception as e:
        logger.warning("❌ FAISS search error: %s", str(e)[:50])
        return []


faiss_search_tool = FunctionTool(func=faiss_search)