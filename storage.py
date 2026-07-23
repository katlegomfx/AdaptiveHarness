import json
import sqlite3
from typing import Dict, List, Optional

DB_PATH = "agent_state.db"


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                session_id TEXT,
                turn_index INTEGER,
                messages TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, turn_index)
            )
        """)
        # 6.4 Long-Term Vector Memory Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_checkpoint(session_id: str, turn_index: int, messages: list) -> None:
    serialized = json.dumps(messages, default=str)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO checkpoints (session_id, turn_index, messages) VALUES (?, ?, ?)",
            (session_id, turn_index, serialized),
        )
        conn.commit()


def load_latest_checkpoint(session_id: str) -> Optional[list]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT messages FROM checkpoints WHERE session_id = ? ORDER BY turn_index DESC LIMIT 1",
            (session_id,),
        )
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None


def list_sessions() -> List[Dict[str, str]]:
    """Retrieves a list of all active sessions and their latest turn metadata."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, MAX(turn_index) as latest_turn, MAX(timestamp) as last_active
            FROM checkpoints
            GROUP BY session_id
            ORDER BY last_active DESC
        """)
        rows = cursor.fetchall()
        return [
            {"session_id": r[0], "latest_turn": r[1], "timestamp": r[2]}
            for r in rows
        ]


def save_learning(text: str, metadata: dict = None):
    """Saves a learning to the long-term memory database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO long_term_memory (text, metadata) VALUES (?, ?)",
                       (text, json.dumps(metadata or {})))
        conn.commit()


def retrieve_learnings(query: str, limit: int = 5) -> list[str]:
    """Retrieves relevant past learnings based on naive keyword overlap."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text FROM long_term_memory ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()

        query_tokens = set(query.lower().split())
        scored = []
        for r in rows:
            text = r[0]
            text_tokens = set(text.lower().split())
            overlap = len(query_tokens.intersection(text_tokens))
            if overlap > 0:
                scored.append((overlap, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:limit]]
