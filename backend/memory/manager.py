# backend/memory/manager.py
"""Persistent memory manager using SQLite with proper locking handling."""
import sqlite3
import hashlib
import time
from datetime import datetime
from pathlib import Path
from config import get_logger

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = 10.0  # SQLite timeout for handling locked databases
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verified_claims (
                    claim_id TEXT PRIMARY KEY,
                    claim_text TEXT UNIQUE NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    evidence_count INTEGER DEFAULT 0,
                    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    processed_input TEXT,
                    verdict TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    execution_time_ms REAL,
                    tool_calls_count INTEGER,
                    evidence_retrieved INTEGER,
                    confidence_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_verified_claims_verdict ON verified_claims(verdict)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_interactions_session ON session_interactions(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_metrics_session ON agent_metrics(session_id)")
            
            conn.commit()
            conn.close()
            logger.warning("💾 Memory database initialized at: %s", self.db_path)
        except Exception as e:
            logger.warning("⚠️  Error initializing database: %s", str(e)[:50])
    
    def _get_conn(self):
        """Get database connection with timeout and WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    
    # ===== SESSION MANAGEMENT =====
    
    def create_session(self, session_id: str, user_id: str = "cli-user"):
        """Create a new session."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id)
            )
            conn.commit()
            conn.close()
            logger.warning("📌 Session created: %s", session_id)
        except sqlite3.IntegrityError:
            logger.warning("📌 Session already exists: %s", session_id)
        except sqlite3.OperationalError as e:
            logger.warning("⚠️  Database locked, retrying: %s", str(e)[:50])
            time.sleep(0.5)
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
                    (session_id, user_id)
                )
                conn.commit()
                conn.close()
            except Exception as e2:
                logger.warning("⚠️  Failed to create session: %s", str(e2)[:50])
        except Exception as e:
            logger.warning("⚠️  Error creating session: %s", str(e)[:50])
    
    def get_session_history(self, session_id: str) -> list:
        """Get all interactions in a session."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT query, verdict, timestamp FROM session_interactions 
                   WHERE session_id = ? ORDER BY timestamp DESC LIMIT 10""",
                (session_id,)
            )
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.warning("⚠️  Error fetching session history: %s", str(e)[:50])
            return []
    
    # ===== CLAIM CACHING =====
    
    def get_cached_verdict(self, claim: str) -> dict:
        """Check if we've already verified this claim."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT * FROM verified_claims 
                   WHERE claim_text = ? 
                   ORDER BY retrieved_at DESC LIMIT 1""",
                (claim,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                logger.warning("✨ Cache hit for claim")
                return dict(row)
            
            logger.warning("📭 Cache miss for claim")
            return None
        except Exception as e:
            logger.warning("⚠️  Error checking cache: %s", str(e)[:50])
            return None
    
    def cache_verdict(self, claim: str, verdict: str, confidence: float = 0.5, 
                     evidence_count: int = 0, session_id: str = None):
        """Cache a verified claim."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            claim_id = hashlib.md5(claim.encode()).hexdigest()
            
            cursor.execute(
                """INSERT INTO verified_claims 
                   (claim_id, claim_text, verdict, confidence, evidence_count, session_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (claim_id, claim, verdict, confidence, evidence_count, session_id)
            )
            conn.commit()
            conn.close()
            logger.warning("💾 Verdict cached for claim")
        except sqlite3.IntegrityError:
            # Claim already exists, update it
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE verified_claims 
                       SET verdict = ?, confidence = ?, evidence_count = ?, session_id = ?
                       WHERE claim_text = ?""",
                    (verdict, confidence, evidence_count, session_id, claim)
                )
                conn.commit()
                conn.close()
                logger.warning("💾 Verdict updated for claim")
            except Exception as e:
                logger.warning("⚠️  Error updating verdict: %s", str(e)[:50])
        except Exception as e:
            logger.warning("⚠️  Error caching verdict: %s", str(e)[:50])
    
    # ===== INTERACTION TRACKING =====
    
    def add_interaction(self, session_id: str, query: str, processed_input: str, verdict: str):
        """Log a user query and result."""
        try:
            # Ensure all parameters are strings
            session_id = str(session_id) if session_id else "unknown"
            query = str(query) if query else ""
            processed_input = str(processed_input) if processed_input else ""
            verdict = str(verdict) if verdict else "UNKNOWN"
            
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO session_interactions 
                   (session_id, query, processed_input, verdict)
                   VALUES (?, ?, ?, ?)""",
                (session_id, query, processed_input, verdict)
            )
            conn.commit()
            conn.close()
            logger.warning("📝 Interaction logged")
        except sqlite3.OperationalError as e:
            logger.warning("⚠️  Database locked: %s", str(e)[:50])
            time.sleep(0.5)
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO session_interactions 
                       (session_id, query, processed_input, verdict)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, query, processed_input, verdict)
                )
                conn.commit()
                conn.close()
            except Exception as e2:
                logger.warning("⚠️  Failed to log interaction: %s", str(e2)[:50])
        except Exception as e:
            logger.warning("⚠️  Error logging interaction: %s", str(e)[:50])
    
    # ===== METRICS TRACKING =====
    
    def record_agent_metric(self, session_id: str, agent_name: str, 
                           execution_time_ms: float, tool_calls: int,
                           evidence_count: int, confidence: float):
        """Record agent execution metrics."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO agent_metrics 
                   (session_id, agent_name, execution_time_ms, tool_calls_count, 
                    evidence_retrieved, confidence_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, agent_name, execution_time_ms, tool_calls, evidence_count, confidence)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("⚠️  Error recording metric: %s", str(e)[:50])
    
    def get_session_metrics(self, session_id: str) -> dict:
        """Get aggregated metrics for a session."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT agent_name, 
                          AVG(execution_time_ms) as avg_time,
                          SUM(tool_calls_count) as total_tools,
                          AVG(confidence_score) as avg_confidence
                   FROM agent_metrics
                   WHERE session_id = ?
                   GROUP BY agent_name""",
                (session_id,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return {
                "agents": [dict(row) for row in rows],
                "session_id": session_id
            }
        except Exception as e:
            logger.warning("⚠️  Error fetching metrics: %s", str(e)[:50])
            return {"agents": [], "session_id": session_id}
    
    def get_all_stats(self) -> dict:
        """Get overall system statistics."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # Total verified claims
            cursor.execute("SELECT COUNT(*) as total FROM verified_claims")
            total_claims = cursor.fetchone()["total"] or 0
            
            # Verdict distribution
            cursor.execute(
                """SELECT verdict, COUNT(*) as count FROM verified_claims 
                   GROUP BY verdict"""
            )
            verdicts = {row["verdict"]: row["count"] for row in cursor.fetchall()}
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) as avg_conf FROM verified_claims")
            result = cursor.fetchone()
            avg_confidence = result["avg_conf"] if result["avg_conf"] else 0.0
            
            # Total sessions
            cursor.execute("SELECT COUNT(*) as total FROM sessions")
            total_sessions = cursor.fetchone()["total"] or 0
            
            conn.close()
            
            return {
                "total_verified_claims": total_claims,
                "verdict_distribution": verdicts or {},
                "average_confidence": float(avg_confidence),
                "total_sessions": total_sessions
            }
        except sqlite3.OperationalError as e:
            logger.warning("⚠️  Database locked when fetching stats: %s", str(e)[:50])
            time.sleep(0.5)
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as total FROM verified_claims")
                total_claims = cursor.fetchone()["total"] or 0
                cursor.execute("""SELECT verdict, COUNT(*) as count FROM verified_claims GROUP BY verdict""")
                verdicts = {row["verdict"]: row["count"] for row in cursor.fetchall()}
                cursor.execute("SELECT AVG(confidence) as avg_conf FROM verified_claims")
                result = cursor.fetchone()
                avg_confidence = result["avg_conf"] if result["avg_conf"] else 0.0
                cursor.execute("SELECT COUNT(*) as total FROM sessions")
                total_sessions = cursor.fetchone()["total"] or 0
                conn.close()
                
                return {
                    "total_verified_claims": total_claims,
                    "verdict_distribution": verdicts or {},
                    "average_confidence": float(avg_confidence),
                    "total_sessions": total_sessions
                }
            except Exception as e2:
                logger.warning("⚠️  Still failed to fetch stats: %s", str(e2)[:50])
                return {
                    "total_verified_claims": 0,
                    "verdict_distribution": {},
                    "average_confidence": 0.0,
                    "total_sessions": 0
                }
        except Exception as e:
            logger.warning("⚠️  Error fetching stats: %s", str(e)[:50])
            return {
                "total_verified_claims": 0,
                "verdict_distribution": {},
                "average_confidence": 0.0,
                "total_sessions": 0
            }