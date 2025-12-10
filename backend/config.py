# ==============================================================================

# FILE 1: backend/config.py (SIMPLIFIED - 90s timeout)
# ======================================================

import os
import logging
import sys
from dotenv import load_dotenv
from datetime import datetime

env_paths = [
    os.path.join(os.path.dirname(__file__), '.env'),
    os.path.join(os.path.dirname(__file__), '..', '.env'),
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found!")
    sys.exit(1)

# FIX: Only set one API key
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ===== RETRY & TIMEOUT CONFIGURATION =====
try:
    from google.genai import types
    
    RETRY_CONFIG = types.HttpRetryOptions(
        initial_delay=2,
        max_delay=30
    )
    
    # INCREASED: 90 seconds (was 60)
    GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        http_options=types.HttpOptions(
            retry_options=RETRY_CONFIG,
            timeout=90.0  # 90 SECONDS
        )
    )
    
    HAS_RETRY_CONFIG = True
    
except Exception as e:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"⚠️ Retry config error: {str(e)[:100]}")
    GENERATE_CONTENT_CONFIG = None
    HAS_RETRY_CONFIG = False

# ===== MODEL SELECTION =====
GEMINI_MODEL_POOL = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.0-flash"
]

ADK_MODEL_NAME = GEMINI_MODEL_POOL[0]
TOP_K = 3

LOG_LEVEL = os.environ.get("FNA_LOG_LEVEL", "WARNING").upper()

_formatter = logging.Formatter("%(levelname)s: %(message)s")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_formatter)

root_logger = logging.getLogger()
if not root_logger.handlers:
    root_logger.addHandler(_handler)
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

def get_logger(name: str):
    return logging.getLogger(name)


# ===== QUOTA MANAGEMENT =====

class QuotaTracker:
    """Track API quota with caching"""
    
    def __init__(self):
        self.api_calls_today = 0
        self.last_minute_reset = None
        self.quota_reset_time = None
        self._update_reset_time()
        self.request_cache = {}
        self.cache_ttl = 3600
        
        logger = get_logger(__name__)
        logger.warning("📊 Quota Tracker initialized")
        logger.warning(f"   Model: {ADK_MODEL_NAME}")
        logger.warning(f"   Timeout: 90 seconds")
        logger.warning(f"   Cache: Enabled (1 hour)")
    
    def _update_reset_time(self):
        now = datetime.now()
        reset_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if reset_time <= now:
            from datetime import timedelta
            reset_time = reset_time + timedelta(days=1)
        self.quota_reset_time = reset_time
    
    def check_quota_available(self, calls_needed: int = 1) -> tuple:
        if datetime.now() > self.quota_reset_time:
            self._update_reset_time()
            self.api_calls_today = 0
        
        remaining = 20 - self.api_calls_today
        
        if remaining < calls_needed:
            hours = (self.quota_reset_time - datetime.now()).total_seconds() / 3600
            return False, f"Quota exhausted. Resets in {hours:.1f}h"
        
        return True, f"✅ {remaining} calls available"
    
    def increment_call_count(self, calls: int = 1):
        self.api_calls_today += calls
    
    def get_request_hash(self, claim: str) -> str:
        import hashlib
        return hashlib.md5(claim[:500].encode()).hexdigest()
    
    def check_cache(self, claim: str) -> dict:
        import time
        req_hash = self.get_request_hash(claim)
        if req_hash in self.request_cache:
            result, ts = self.request_cache[req_hash]
            if time.time() - ts < self.cache_ttl:
                return result
            else:
                del self.request_cache[req_hash]
        return None
    
    def store_cache(self, claim: str, result: dict):
        import time
        req_hash = self.get_request_hash(claim)
        self.request_cache[req_hash] = (result, time.time())
    
    def get_stats(self) -> dict:
        return {
            "calls_today": self.api_calls_today,
            "remaining": max(0, 20 - self.api_calls_today),
            "timeout_seconds": 90,
            "cache_enabled": True
        }

quota_tracker = QuotaTracker()