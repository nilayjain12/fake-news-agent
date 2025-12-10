# backend/config.py (FIXED - Compatible with current google-genai)
"""
Configuration with built-in quota management and intelligent caching
Uses compatible HttpRetryOptions from google.genai.types
"""
import os
import logging
import sys
from dotenv import load_dotenv
from datetime import datetime

env_paths = [
    os.path.join(os.path.dirname(__file__), '.env'),
    os.path.join(os.path.dirname(__file__), '..', '.env'),
]

loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        loaded = True
        break

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in environment!")
    sys.exit(1)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ===== RETRY CONFIGURATION (For 429 Errors) =====
# Using only compatible parameters with google.genai.types
try:
    from google.genai import types
    
    RETRY_CONFIG = types.HttpRetryOptions(
        initial_delay=2,      # Start with 2 second delay
        max_delay=30          # Max 30 seconds between retries
        # Note: backoff_multiplier not supported in current version
        # Exponential backoff is automatic in google-genai
    )
    
    # ===== GENERATE CONTENT CONFIG =====
    GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        http_options=types.HttpOptions(
            retry_options=RETRY_CONFIG,
            timeout=30.0  # 30 second timeout per request
        )
    )
    
    HAS_RETRY_CONFIG = True
except Exception as e:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"⚠️  Could not load retry config: {str(e)[:100]}")
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

def get_logger(name: str):
    return logging.getLogger(name)


# ===== QUOTA MANAGEMENT (429 Prevention) =====

class QuotaTracker:
    """
    Track API quota usage to prevent 429 errors.
    
    FREE TIER LIMITS:
    - 20 requests/day (RPD)
    - 10 requests/minute (RPM)
    
    STRATEGY:
    1. Cache identical requests (deduplication)
    2. Track daily usage and pace requests
    3. Use built-in retry for transient failures
    4. Store request hashes to avoid duplicate processing
    """
    
    def __init__(self):
        self.api_calls_today = 0
        self.api_calls_this_minute = 0
        self.last_minute_reset = None
        self.quota_reset_time = None
        self._update_reset_time()
        
        # Request deduplication cache
        self.request_cache = {}  # hash -> (result, timestamp)
        self.cache_ttl = 3600  # Cache for 1 hour
        
        logger = get_logger(__name__)
        logger.warning("📊 Quota Tracker initialized (429-Safe)")
        logger.warning(f"   Model: {ADK_MODEL_NAME}")
        logger.warning(f"   Free Tier: 20 RPD (requests/day), 10 RPM (requests/minute)")
        logger.warning(f"   Retry: Built-in exponential backoff (via google-genai)")
        logger.warning(f"   Cache: Request deduplication enabled (1 hour TTL)")
        
        if HAS_RETRY_CONFIG:
            logger.warning(f"   ✅ Retry config loaded successfully")
        else:
            logger.warning(f"   ⚠️  Retry config unavailable - using fallback")
    
    def _update_reset_time(self):
        """Calculate next quota reset (midnight PT / 8am UTC)"""
        now = datetime.now()
        reset_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if reset_time <= now:
            from datetime import timedelta
            reset_time = reset_time + timedelta(days=1)
        
        self.quota_reset_time = reset_time
        return reset_time
    
    def check_quota_available(self, calls_needed: int = 1) -> tuple:
        """
        Check if quota available before making API calls.
        Returns: (allowed: bool, message: str)
        """
        logger = get_logger(__name__)
        
        # Check daily reset
        if datetime.now() > self.quota_reset_time:
            logger.warning("🔄 Daily quota reset")
            self._update_reset_time()
            self.api_calls_today = 0
            self.api_calls_this_minute = 0
        
        # Reset minute counter if needed
        if self.last_minute_reset is None or (datetime.now() - self.last_minute_reset).total_seconds() >= 60:
            self.api_calls_this_minute = 0
            self.last_minute_reset = datetime.now()
        
        remaining_daily = 20 - self.api_calls_today  # Hard limit: 20/day
        remaining_minute = 10 - self.api_calls_this_minute  # Hard limit: 10/min
        
        # Check daily limit (with buffer for safety)
        if remaining_daily < calls_needed:
            time_to_reset = self.quota_reset_time - datetime.now()
            hours = time_to_reset.total_seconds() / 3600
            logger.warning(f"🚫 Daily quota EXHAUSTED ({self.api_calls_today}/20 used)")
            logger.warning(f"   Resets in {hours:.1f} hours")
            return False, f"Daily quota exhausted (20/day used). Resets in {hours:.1f}h."
        
        # Check minute limit
        if remaining_minute < calls_needed:
            logger.warning(f"⏱️ Minute rate limit hit ({self.api_calls_this_minute}/10)")
            return False, "Rate limit hit (10/min). Please wait 60 seconds..."
        
        # Warnings for low quota
        if remaining_daily <= 2:
            logger.warning(f"🚨 CRITICAL: Only {remaining_daily} API calls remaining!")
        elif remaining_daily <= 5:
            logger.warning(f"⚠️ WARNING: Only {remaining_daily} API calls remaining")
        else:
            logger.warning(f"✅ Quota OK: {remaining_daily} API calls available")
        
        return True, f"✅ {remaining_daily} requests remaining (daily), {remaining_minute} (minute)"
    
    def increment_call_count(self, calls: int = 1):
        """Record an API call"""
        self.api_calls_today += calls
        self.api_calls_this_minute += calls
        logger = get_logger(__name__)
        logger.warning(f"📈 API Call recorded: {self.api_calls_today}/20 used today")
    
    def get_request_hash(self, claim: str) -> str:
        """Generate hash for request deduplication"""
        import hashlib
        return hashlib.md5(claim[:500].encode()).hexdigest()
    
    def check_cache(self, claim: str) -> dict:
        """Check if we've processed this claim recently"""
        import time
        req_hash = self.get_request_hash(claim)
        
        if req_hash in self.request_cache:
            cached_result, timestamp = self.request_cache[req_hash]
            age = time.time() - timestamp
            
            if age < self.cache_ttl:
                logger = get_logger(__name__)
                logger.warning(f"💾 Cache HIT (age: {age:.0f}s) - Saving API calls!")
                return cached_result
            else:
                # Cache expired
                del self.request_cache[req_hash]
        
        return None
    
    def store_cache(self, claim: str, result: dict):
        """Store result in cache"""
        import time
        req_hash = self.get_request_hash(claim)
        self.request_cache[req_hash] = (result, time.time())
        logger = get_logger(__name__)
        logger.warning(f"💾 Result cached for 1 hour")
    
    def get_stats(self) -> dict:
        """Get quota statistics"""
        self.check_quota_available()
        return {
            "model": ADK_MODEL_NAME,
            "calls_today": self.api_calls_today,
            "daily_limit": 20,
            "remaining": max(0, 20 - self.api_calls_today),
            "minute_limit": 10,
            "remaining_minute": max(0, 10 - self.api_calls_this_minute),
            "quota_reset_time": self.quota_reset_time.isoformat(),
            "retry_strategy": "Built-in exponential backoff (google-genai)",
            "cache_enabled": True,
            "cache_hits": sum(1 for _ in self.request_cache),
            "status": "🟢 healthy" if self.api_calls_today < 18 else "🟡 caution" if self.api_calls_today < 20 else "🔴 exhausted"
        }


# Global quota tracker
quota_tracker = QuotaTracker()