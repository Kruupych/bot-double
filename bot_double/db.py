from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER NOT NULL UNIQUE,
    username TEXT,
    first_name TEXT,
    last_name TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_user_chat_timestamp
    ON messages(user_id, chat_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS chat_settings (
    chat_id INTEGER PRIMARY KEY,
    auto_imitate_enabled INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    def __init__(self, path: Path, max_messages_per_user: int) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._max_messages_per_user = max_messages_per_user
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- user helpers -----------------------------------------------------------------
    def upsert_user(self, telegram_id: int, username: Optional[str], first_name: Optional[str], last_name: Optional[str]) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "SELECT id, username, first_name, last_name FROM users WHERE telegram_id = ?",
                (telegram_id,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor = self._conn.execute(
                    "INSERT INTO users (telegram_id, username, first_name, last_name) VALUES (?, ?, ?, ?)",
                    (telegram_id, username, first_name, last_name),
                )
                return cursor.lastrowid

            updates = []
            params: List[object] = []
            if username and row["username"] != username:
                updates.append("username = ?")
                params.append(username)
            if first_name and row["first_name"] != first_name:
                updates.append("first_name = ?")
                params.append(first_name)
            if last_name and row["last_name"] != last_name:
                updates.append("last_name = ?")
                params.append(last_name)
            if updates:
                params.append(telegram_id)
                self._conn.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE telegram_id = ?",
                    tuple(params),
                )
            return int(row["id"])

    # --- message helpers --------------------------------------------------------------
    def store_message(self, chat_id: int, user_id: int, text: str, timestamp: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (chat_id, user_id, text, timestamp) VALUES (?, ?, ?, ?)",
                (chat_id, user_id, text, timestamp),
            )
            overflow = self._count_messages(chat_id, user_id) - self._max_messages_per_user
            if overflow > 0:
                self._trim_messages(chat_id, user_id, overflow)

    def _count_messages(self, chat_id: int, user_id: int) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        row = cursor.fetchone()
        return int(row["cnt"]) if row else 0

    def _trim_messages(self, chat_id: int, user_id: int, excess: int) -> None:
        self._conn.execute(
            """
            DELETE FROM messages
            WHERE id IN (
                SELECT id FROM messages
                WHERE chat_id = ? AND user_id = ?
                ORDER BY timestamp ASC, id ASC
                LIMIT ?
            )
            """,
            (chat_id, user_id, excess),
        )

    def get_random_messages(self, chat_id: int, user_id: int, limit: int) -> List[str]:
        cursor = self._conn.execute(
            "SELECT text FROM messages WHERE chat_id = ? AND user_id = ? ORDER BY RANDOM() LIMIT ?",
            (chat_id, user_id, limit),
        )
        return [row["text"] for row in cursor.fetchall()]

    def get_message_count(self, chat_id: int, user_id: int) -> int:
        return self._count_messages(chat_id, user_id)

    def get_user_by_username(self, username: str) -> Optional[sqlite3.Row]:
        cursor = self._conn.execute(
            "SELECT id, telegram_id, username, first_name, last_name FROM users WHERE username = ?",
            (username,),
        )
        return cursor.fetchone()

    def get_user_by_telegram_id(self, telegram_id: int) -> Optional[sqlite3.Row]:
        cursor = self._conn.execute(
            "SELECT id, telegram_id, username, first_name, last_name FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        return cursor.fetchone()

    # --- profiles ---------------------------------------------------------------------
    def get_profiles(self, chat_id: int) -> List[sqlite3.Row]:
        cursor = self._conn.execute(
            """
            SELECT u.username, u.first_name, u.last_name, COUNT(m.id) as message_count
            FROM users u
            JOIN messages m ON m.user_id = u.id
            WHERE m.chat_id = ?
            GROUP BY u.id
            ORDER BY message_count DESC
            """,
            (chat_id,),
        )
        return cursor.fetchall()

    def get_recent_messages_for_user(
        self, chat_id: int, user_id: int, limit: int
    ) -> List[str]:
        if limit <= 0:
            return []
        cursor = self._conn.execute(
            """
            SELECT text FROM messages
            WHERE chat_id = ? AND user_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (chat_id, user_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return [row["text"] for row in rows]

    # --- chat settings ----------------------------------------------------------------
    def set_auto_imitate(self, chat_id: int, enabled: bool) -> None:
        value = 1 if enabled else 0
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO chat_settings (chat_id, auto_imitate_enabled) VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET auto_imitate_enabled = excluded.auto_imitate_enabled
                """,
                (chat_id, value),
            )

    def is_auto_imitate_enabled(self, chat_id: int) -> bool:
        cursor = self._conn.execute(
            "SELECT auto_imitate_enabled FROM chat_settings WHERE chat_id = ?",
            (chat_id,),
        )
        row = cursor.fetchone()
        return bool(row[0]) if row else False

    def get_recent_chat_messages(self, chat_id: int, limit: int) -> List[sqlite3.Row]:
        cursor = self._conn.execute(
            """
            SELECT u.username, u.first_name, u.last_name, m.text
            FROM messages m
            JOIN users u ON u.id = m.user_id
            WHERE m.chat_id = ?
            ORDER BY m.timestamp DESC, m.id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return rows

    def get_top_participants(
        self, chat_id: int, exclude_user_id: Optional[int], limit: int
    ) -> List[sqlite3.Row]:
        if limit <= 0:
            return []
        params: List[object] = [chat_id]
        exclusion_clause = ""
        if exclude_user_id is not None:
            exclusion_clause = "AND u.id != ?"
            params.append(exclude_user_id)
        cursor = self._conn.execute(
            f"""
            SELECT u.id, u.username, u.first_name, u.last_name, COUNT(m.id) as message_count
            FROM users u
            JOIN messages m ON m.user_id = u.id
            WHERE m.chat_id = ? {exclusion_clause}
            GROUP BY u.id
            ORDER BY message_count DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return cursor.fetchall()

    def delete_user_data(self, telegram_id: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "SELECT id FROM users WHERE telegram_id = ?",
                (telegram_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            user_id = int(row["id"])
            self._conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return True
