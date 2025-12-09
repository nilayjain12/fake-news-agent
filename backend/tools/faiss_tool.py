# ==============================================================================
# FILE 1: backend/tools/faiss_tool.py (REPLACE EXISTING)
# ==============================================================================

"""
FAISS Search Tool - Pure ADK FunctionTool
"""
from google.adk.tools import FunctionTool
from typing import List, Dict, Optional
from config import FAISS_INDEX_PATH, TOP_K, get_logger, EMBEDDING_MODEL
import datetime
import os

logger = get_logger(__name__)

# Lazy-loaded FAISS database
_db = None
_load_error = None


def _load_faiss():
    """Lazy load FAISS index"""
    global _db, _load_error
    
    if _load_error is not None:
        raise _load_error
    
    if _db is None:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            
            _db = FAISS.load_local(
                folder_path=FAISS_INDEX_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            logger.warning("✅ FAISS index loaded")
        except Exception as e:
            _load_error = e
            logger.warning(f"❌ FAISS load error: {str(e)[:100]}")
            raise
    
    return _db


def search_faiss_knowledge_base(
    query: str,
    k: int = TOP_K
) -> List[Dict]:
    """
    Search FAISS knowledge base for fact-checking information.
    
    This tool searches a semantic vector database containing verified facts,
    historical information, and established knowledge. Best for finding
    background context and previously verified information.
    
    Args:
        query: The claim or question to search for
        k: Number of results to return (default: 5)
    
    Returns:
        List of dictionaries with keys:
        - rank: Result ranking (1-based)
        - content: Document text content
        - source: Source identifier
        - metadata: Additional metadata
        - age_days: Age in days (if date available)
        - is_old_source: Boolean flag for old data (>90 days)
        - _source: Always "faiss"
    
    Note: Results may not reflect very recent events. Use google_search for current info.
    """
    logger.warning(f"🔎 FAISS Tool: Searching for '{query[:60]}'")
    
    try:
        db = _load_faiss()
        docs = db.similarity_search(query, k=k)
        
        results = []
        current_time = datetime.datetime.now()
        
        for i, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            
            # Calculate age if date available
            doc_date = meta.get("date")
            age_days = None
            is_old = False
            
            if doc_date:
                try:
                    doc_datetime = datetime.datetime.fromisoformat(str(doc_date))
                    age_days = (current_time - doc_datetime).days
                    is_old = age_days > 90
                except:
                    pass
            
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
                "age_days": age_days,
                "is_old_source": is_old,
                "_source": "faiss"
            }
            
            results.append(result)
        
        logger.warning(f"   ✅ Found {len(results)} FAISS results")
        return results
        
    except Exception as e:
        logger.warning(f"❌ FAISS search error: {str(e)[:100]}")
        return []


# Create ADK FunctionTool (correct API)
faiss_search_tool = FunctionTool(search_faiss_knowledge_base)