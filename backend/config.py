# backend/config.py (CORRECTED - Use Right Model)
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

# ===== CORRECT MODEL SELECTION =====
# IMPORTANT: Use gemini-2.5-flash-lite for text operations
# This model has better free tier limits:
# - 10 RPM (vs 5 for flash)
# - 20 RPD (same as flash but optimized for high-frequency)
# - Better for batch operations and minimal API calls
# ===== GEMINI MODEL FAILOVER CHAIN =====

GEMINI_MODEL_POOL = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.0-flash"
]

ADK_MODEL_NAME = GEMINI_MODEL_POOL[0]  # Primary model

# Alternative if flash-lite doesn't work:
# ADK_MODEL_NAME = "gemini-2.5-flash"

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


# ===== API QUOTA MANAGEMENT =====

# Free tier limits for gemini-2.5-flash-lite
# Based on actual quota shown in Google AI Studio
FREE_TIER_RPM = 10          # Requests per minute (flash-lite can handle 10)
FREE_TIER_RPD = 20          # Requests per day (20 total)
FREE_TIER_TPM = 250000      # Tokens per minute

# ⚠️  CRITICAL: With 20 RPD limit and 2 API calls per query:
# Maximum queries per day = 20 / 2 = 10 queries
# This matches our optimized pipeline's capability
TARGET_API_CALLS_PER_QUERY = 2
TARGET_QUERIES_PER_DAY = 10  # 10 queries × 2 calls = 20 total (at limit)
TARGET_DAILY_API_CALLS = TARGET_API_CALLS_PER_QUERY * TARGET_QUERIES_PER_DAY

class QuotaTracker:
    """Track API quota usage to prevent 429 errors"""
    
    def __init__(self):
        self.api_calls_today = 0
        self.api_calls_this_minute = 0
        self.last_minute_reset = None
        self.quota_reset_time = None
        self._update_reset_time()
        logger = get_logger(__name__)
        logger.warning(f"📊 Quota Tracker initialized")
        logger.warning(f"   Model: {ADK_MODEL_NAME}")
        logger.warning(f"   Free Tier Limits: {FREE_TIER_RPM} RPM, {FREE_TIER_RPD} RPD")
        logger.warning(f"   Daily Capacity: {TARGET_QUERIES_PER_DAY} queries × {TARGET_API_CALLS_PER_QUERY} calls = {TARGET_DAILY_API_CALLS} API calls")
    
    def _update_reset_time(self):
        """Calculate next quota reset (midnight PT / 8am UTC)"""
        now = datetime.now()
        # Google resets quotas at midnight PT (8am UTC during daylight saving)
        reset_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if reset_time <= now:
            # If reset time has passed, next reset is tomorrow
            from datetime import timedelta
            reset_time = reset_time + timedelta(days=1)
        
        self.quota_reset_time = reset_time
        return reset_time
    
    def check_quota_available(self, calls_needed: int = 1) -> tuple:
        """Check if we have quota for the API calls"""
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
        
        remaining_daily = FREE_TIER_RPD - self.api_calls_today
        remaining_minute = FREE_TIER_RPM - self.api_calls_this_minute
        
        # Check daily limit
        if remaining_daily < calls_needed:
            time_to_reset = self.quota_reset_time - datetime.now()
            hours = time_to_reset.total_seconds() / 3600
            return False, f"❌ Daily quota limit ({FREE_TIER_RPD}) reached. Resets in {hours:.1f} hours"
        
        # Check minute limit
        if remaining_minute < calls_needed:
            return False, f"❌ Minute rate limit ({FREE_TIER_RPM}/min) reached. Wait before retrying."
        
        # Warnings for low quota
        if remaining_daily <= 2:
            return True, f"⚠️  CRITICAL: Only {remaining_daily} API calls remaining today!"
        elif remaining_daily <= 5:
            return True, f"⚠️  WARNING: Only {remaining_daily} API calls remaining today"
        else:
            return True, f"✅ {remaining_daily} API calls available today"
    
    def increment_call_count(self, calls: int = 1) -> int:
        """Increment API call count"""
        self.api_calls_today += calls
        self.api_calls_this_minute += calls
        return self.api_calls_today
    
    def get_stats(self) -> dict:
        """Get quota statistics"""
        self.check_quota_available()
        return {
            "model": ADK_MODEL_NAME,
            "calls_today": self.api_calls_today,
            "daily_limit": FREE_TIER_RPD,
            "remaining": max(0, FREE_TIER_RPD - self.api_calls_today),
            "minute_limit": FREE_TIER_RPM,
            "remaining_minute": max(0, FREE_TIER_RPM - self.api_calls_this_minute),
            "quota_reset_time": self.quota_reset_time.isoformat(),
            "target_daily_calls": TARGET_DAILY_API_CALLS,
            "status": "🟢 healthy" if self.api_calls_today <= TARGET_DAILY_API_CALLS else "🔴 at risk",
            "daily_capacity": f"{self.api_calls_today}/{TARGET_DAILY_API_CALLS}",
            "strategy": "Ultra-optimized: 2 calls/query, cache-first"
        }

# Global quota tracker
quota_tracker = QuotaTracker()