# File: backend/memory/schema.py
"""Database schema for persistent memory."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS verified_claims (
    claim_id TEXT PRIMARY KEY,
    claim_text TEXT UNIQUE NOT NULL,
    verdict TEXT NOT NULL,  -- "Likely True", "Likely False", "Unverified"
    confidence REAL NOT NULL,  -- 0.0 to 1.0
    evidence_count INTEGER,
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS session_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    processed_input TEXT,
    verdict TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

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
);

CREATE INDEX IF NOT EXISTS idx_verified_claims_verdict ON verified_claims(verdict);
CREATE INDEX IF NOT EXISTS idx_session_interactions_session ON session_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_session ON agent_metrics(session_id);
"""