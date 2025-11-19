"""Defining a Retriever class to fetch relevant documents from a FAISS index."""
from config import TOP_K, get_logger
from loader.embeddings_loader import load_faiss_index

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        logger.info("Initializing Retriever and loading FAISS DB")
        self.db = load_faiss_index()

    def fetch_context(self, query: str) -> str:
        """Retrieve top-matching documents as context."""
        logger.debug("Fetching context for query: %s", query)
        docs = self.db.similarity_search(query, k=TOP_K)
        logger.info("Retrieved %d documents for query", len(docs) if docs else 0)
        return "\n\n".join([doc.page_content for doc in docs])