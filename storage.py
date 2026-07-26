# storage.py
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                metadata TEXT,
                embedding TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_text TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                source TEXT DEFAULT 'user',
                attempts INTEGER DEFAULT 0,
                notes TEXT,
                parent_id INTEGER,
                embedding TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_attempted_at DATETIME,
                FOREIGN KEY(parent_id) REFERENCES long_term_goals(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_improvements_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                description TEXT,
                status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_checkpoint(session_id: str, turn_index: int, messages: list) -> None:
    serialized = json.dumps(messages, default=str)
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute(
            "INSERT OR REPLACE INTO checkpoints (session_id, turn_index, messages) VALUES (?, ?, ?)",
            (session_id, turn_index, serialized),
        )
        conn.commit()


def load_latest_checkpoint(session_id: str) -> Optional[list]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT messages FROM checkpoints WHERE session_id = ? ORDER BY turn_index DESC LIMIT 1", (session_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None


def list_sessions() -> List[Dict[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, MAX(turn_index) as latest_turn, MAX(timestamp) as last_active
            FROM checkpoints GROUP BY session_id ORDER BY last_active DESC
        """)
        return [{"session_id": r[0], "latest_turn": r[1], "timestamp": r[2]} for r in cursor.fetchall()]


def save_learning(text: str, metadata: dict = None, embedding: list = None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute("INSERT INTO long_term_memory (text, metadata, embedding) VALUES (?, ?, ?)",
                              (text, json.dumps(metadata or {}), json.dumps(embedding or [])))
        conn.commit()


def retrieve_learnings(query: str, query_emb: list = None, limit: int = 5) -> list[str]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text, embedding FROM long_term_memory ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()

        if query_emb:
            scored = []
            for r in rows:
                text, emb_str = r[0], r[1]
                emb = json.loads(emb_str) if emb_str else []
                from llm_backend import cosine_similarity
                sim = cosine_similarity(query_emb, emb)
                scored.append((sim, text))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [text for score, text in scored[:limit] if score > 0.1]
        else:
            # Fallback to naive overlap
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


def add_long_term_goal(goal_text: str, priority: int = 5, source: str = "user", parent_id: int = None, embedding: list = None) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO long_term_goals (goal_text, priority, source, parent_id, embedding) VALUES (?, ?, ?, ?, ?)",
            (goal_text, priority, source, parent_id, json.dumps(embedding or [])),
        )
        conn.commit()
        return cur.lastrowid


def get_next_long_term_goal() -> Optional[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, goal_text, priority, attempts, embedding
            FROM long_term_goals
            WHERE status IN ('pending', 'in_progress')
            ORDER BY priority DESC,
                     CASE WHEN last_attempted_at IS NULL THEN 0 ELSE 1 END,
                     last_attempted_at ASC,
                     created_at ASC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "goal_text": row[1], "priority": row[2], "attempts": row[3], "embedding": json.loads(row[4]) if row[4] else []}


def list_long_term_goals(status: Optional[str] = None) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        if status:
            cur.execute(
                "SELECT id, goal_text, priority, status, attempts, created_at, parent_id "
                "FROM long_term_goals WHERE status = ? "
                "ORDER BY priority DESC, created_at ASC", (status,))
        else:
            cur.execute(
                "SELECT id, goal_text, priority, status, attempts, created_at, parent_id "
                "FROM long_term_goals ORDER BY priority DESC, created_at ASC")
        return [
            {"id": r[0], "goal_text": r[1], "priority": r[2],
             "status": r[3], "attempts": r[4], "created_at": r[5], "parent_id": r[6]}
            for r in cur.fetchall()
        ]


def mark_goal_attempted(goal_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute("""
            UPDATE long_term_goals
               SET attempts = attempts + 1,
                   status = 'in_progress',
                   last_attempted_at = CURRENT_TIMESTAMP
             WHERE id = ?
        """, (goal_id,))
        conn.commit()


def mark_goal_completed(goal_id: int, notes: str = "") -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute("""
            UPDATE long_term_goals
               SET status = 'completed', notes = ?, last_attempted_at = CURRENT_TIMESTAMP
             WHERE id = ?
        """, (notes, goal_id))
        conn.commit()


def mark_goal_blocked(goal_id: int, reason: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute("""
            UPDATE long_term_goals
               SET status = 'blocked', notes = ?, last_attempted_at = CURRENT_TIMESTAMP
             WHERE id = ?
        """, (reason, goal_id))
        conn.commit()


def mark_goal_decomposed(goal_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute("""
            UPDATE long_term_goals SET status = 'decomposed' WHERE id = ?
        """, (goal_id,))
        conn.commit()


def update_goal_embedding(goal_id: int, embedding: list) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute("UPDATE long_term_goals SET embedding = ? WHERE id = ?",
                              (json.dumps(embedding), goal_id))
        conn.commit()


def log_system_improvement(file_path: str, description: str, status: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute(
            "INSERT INTO system_improvements_log (file_path, description, status) VALUES (?, ?, ?)",
            (file_path, description, status))
        conn.commit()
