from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple
import json
import time

from .utils import normalize_alias

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
    context_only INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_user_chat_timestamp
    ON messages(user_id, chat_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS chat_settings (
    chat_id INTEGER PRIMARY KEY,
    auto_imitate_enabled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pair_interactions (
    chat_id INTEGER NOT NULL,
    speaker_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    total_count INTEGER NOT NULL DEFAULT 0,
    informal_count INTEGER NOT NULL DEFAULT 0,
    formal_count INTEGER NOT NULL DEFAULT 0,
    teasing_count INTEGER NOT NULL DEFAULT 0,
    sample_messages TEXT,
    pending_messages INTEGER NOT NULL DEFAULT 0,
    last_analyzed_at INTEGER,
    analysis_summary TEXT,
    analysis_details TEXT,
    PRIMARY KEY (chat_id, speaker_id, target_id),
    FOREIGN KEY(speaker_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(target_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pair_interaction_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    speaker_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    FOREIGN KEY(speaker_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(target_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pair_messages_lookup
    ON pair_interaction_messages(chat_id, speaker_id, target_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS persona_profiles (
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    summary TEXT,
    details TEXT,
    pending_messages INTEGER NOT NULL DEFAULT 0,
    last_analyzed_at INTEGER,
    PRIMARY KEY (chat_id, user_id),
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_aliases (
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (chat_id, normalized_alias),
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS persona_preferences (
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    mode INTEGER NOT NULL DEFAULT 2, -- 0=summary,1=card,2=auto
    PRIMARY KEY (chat_id, user_id),
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
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
        self._ensure_schema()

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
    def store_message(
        self, chat_id: int, user_id: int, text: str, timestamp: int, *, context_only: bool = False
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO messages (chat_id, user_id, text, timestamp, context_only)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, user_id, text, timestamp, 1 if context_only else 0),
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
            """
            SELECT text FROM messages
            WHERE chat_id = ? AND user_id = ? AND context_only = 0
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (chat_id, user_id, limit),
        )
        return [row["text"] for row in cursor.fetchall()]

    def get_message_count(self, chat_id: int, user_id: int) -> int:
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) as cnt FROM messages
            WHERE chat_id = ? AND user_id = ? AND context_only = 0
            """,
            (chat_id, user_id),
        )
        row = cursor.fetchone()
        return int(row["cnt"]) if row else 0

    # --- persona preferences ---------------------------------------------------------
    def set_persona_preference(self, chat_id: int, user_id: int, mode: int) -> None:
        if mode not in (0, 1, 2, 3):
            raise ValueError("mode must be 0 (summary), 1 (card), 2 (auto), or 3 (combined)")
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO persona_preferences (chat_id, user_id, mode)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id, user_id)
                DO UPDATE SET mode = excluded.mode
                """,
                (chat_id, user_id, mode),
            )

    def get_persona_preference(self, chat_id: int, user_id: int) -> int:
        cursor = self._conn.execute(
            "SELECT mode FROM persona_preferences WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        row = cursor.fetchone()
        return int(row["mode"]) if row and row["mode"] is not None else 2

    def get_user_by_username(self, username: str) -> Optional[sqlite3.Row]:
        cursor = self._conn.execute(
            """
            SELECT id, telegram_id, username, first_name, last_name
            FROM users
            WHERE LOWER(username) = LOWER(?)
            """,
            (username,),
        )
        return cursor.fetchone()

    def get_chat_participants(self, chat_id: int) -> List[sqlite3.Row]:
        cursor = self._conn.execute(
            """
            SELECT DISTINCT u.id, u.telegram_id, u.username, u.first_name, u.last_name
            FROM users u
            JOIN messages m ON m.user_id = u.id
            WHERE m.chat_id = ?
            """,
            (chat_id,),
        )
        return cursor.fetchall()

    def add_aliases(
        self, chat_id: int, user_id: int, aliases: List[str]
    ) -> Tuple[List[str], List[str]]:
        if not aliases:
            return [], []
        added: List[str] = []
        skipped: List[str] = []
        timestamp = int(time.time())
        with self._lock, self._conn:
            for alias in aliases:
                normalized = normalize_alias(alias)
                if not normalized:
                    skipped.append(alias)
                    continue
                existing = self._conn.execute(
                    """
                    SELECT user_id
                    FROM user_aliases
                    WHERE chat_id = ? AND normalized_alias = ?
                    """,
                    (chat_id, normalized),
                ).fetchone()
                if existing is not None:
                    current_user_id = int(existing["user_id"])
                    if current_user_id == user_id:
                        skipped.append(alias)
                        continue
                    # Alias принадлежал другому пользователю — перекидываем
                    self._conn.execute(
                        """
                        UPDATE user_aliases
                        SET user_id = ?, alias = ?, created_at = ?
                        WHERE chat_id = ? AND normalized_alias = ?
                        """,
                        (
                            user_id,
                            alias.strip(),
                            timestamp,
                            chat_id,
                            normalized,
                        ),
                    )
                    added.append(alias)
                    continue
                try:
                    self._conn.execute(
                        """
                        INSERT INTO user_aliases (chat_id, user_id, alias, normalized_alias, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (chat_id, user_id, alias.strip(), normalized, timestamp),
                    )
                except sqlite3.IntegrityError:
                    skipped.append(alias)
                else:
                    added.append(alias)
        return added, skipped

    def delete_aliases(self, chat_id: int, user_id: int) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM user_aliases WHERE chat_id = ? AND user_id = ?",
                (chat_id, user_id),
            )
        return cursor.rowcount

    def get_aliases_for_chat(self, chat_id: int) -> List[sqlite3.Row]:
        cursor = self._conn.execute(
            """
            SELECT alias, normalized_alias, user_id
            FROM user_aliases
            WHERE chat_id = ?
            """,
            (chat_id,),
        )
        return cursor.fetchall()

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
            WHERE m.chat_id = ? AND m.context_only = 0
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
            WHERE chat_id = ? AND user_id = ? AND context_only = 0
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (chat_id, user_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return [row["text"] for row in rows]

    def get_recent_messages_with_timestamp(
        self, chat_id: int, user_id: int, limit: int
    ) -> List[sqlite3.Row]:
        if limit <= 0:
            return []
        cursor = self._conn.execute(
            """
            SELECT text, timestamp
            FROM messages
            WHERE chat_id = ? AND user_id = ? AND context_only = 0
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (chat_id, user_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return rows

    def update_pair_stats(
        self,
        chat_id: int,
        speaker_id: int,
        target_id: int,
        *,
        informal: bool,
        formal: bool,
        teasing: bool,
        sample_text: Optional[str],
        full_text: Optional[str],
        timestamp: int,
    ) -> None:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                SELECT total_count, informal_count, formal_count, teasing_count, sample_messages, pending_messages
                FROM pair_interactions
                WHERE chat_id = ? AND speaker_id = ? AND target_id = ?
                """,
                (chat_id, speaker_id, target_id),
            )
            row = cursor.fetchone()
            total_count = int(row["total_count"]) if row else 0
            informal_count = int(row["informal_count"]) if row else 0
            formal_count = int(row["formal_count"]) if row else 0
            teasing_count = int(row["teasing_count"]) if row else 0
            samples = self._decode_samples(row["sample_messages"]) if row else []
            pending_messages = (
                int(row["pending_messages"])
                if row and row["pending_messages"] is not None
                else 0
            )

            total_count += 1
            if informal:
                informal_count += 1
            if formal:
                formal_count += 1
            if teasing:
                teasing_count += 1
            pending_messages += 1

            normalized_sample = self._normalize_sample(sample_text)
            if normalized_sample:
                if normalized_sample in samples:
                    samples.remove(normalized_sample)
                samples.append(normalized_sample)
                samples = samples[-5:]

            if full_text:
                self._conn.execute(
                    """
                    INSERT INTO pair_interaction_messages (
                        chat_id,
                        speaker_id,
                        target_id,
                        text,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (chat_id, speaker_id, target_id, full_text, timestamp),
                )
                self._conn.execute(
                    """
                    DELETE FROM pair_interaction_messages
                    WHERE id IN (
                        SELECT id FROM pair_interaction_messages
                        WHERE chat_id = ? AND speaker_id = ? AND target_id = ?
                        ORDER BY timestamp DESC, id DESC
                        LIMIT -1 OFFSET 100
                    )
                    """,
                    (chat_id, speaker_id, target_id),
                )

            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO pair_interactions (
                        chat_id,
                        speaker_id,
                        target_id,
                        total_count,
                        informal_count,
                        formal_count,
                        teasing_count,
                        sample_messages,
                        pending_messages,
                        last_analyzed_at,
                        analysis_summary,
                        analysis_details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        speaker_id,
                        target_id,
                        total_count,
                        informal_count,
                        formal_count,
                        teasing_count,
                        self._encode_samples(samples),
                        pending_messages,
                        None,
                        None,
                        None,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE pair_interactions
                    SET total_count = ?,
                        informal_count = ?,
                        formal_count = ?,
                        teasing_count = ?,
                        sample_messages = ?,
                        pending_messages = ?
                    WHERE chat_id = ? AND speaker_id = ? AND target_id = ?
                    """,
                    (
                        total_count,
                        informal_count,
                        formal_count,
                        teasing_count,
                        self._encode_samples(samples),
                        pending_messages,
                        chat_id,
                        speaker_id,
                        target_id,
                    ),
                )

    def get_pair_stats(
        self, chat_id: int, speaker_id: int, target_id: int
    ) -> Optional[dict[str, object]]:
        cursor = self._conn.execute(
            """
            SELECT total_count,
                   informal_count,
                   formal_count,
                   teasing_count,
                   sample_messages,
                   pending_messages,
                   last_analyzed_at,
                   analysis_summary,
                   analysis_details
            FROM pair_interactions
            WHERE chat_id = ? AND speaker_id = ? AND target_id = ?
            """,
            (chat_id, speaker_id, target_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "total_count": int(row["total_count"]),
            "informal_count": int(row["informal_count"]),
            "formal_count": int(row["formal_count"]),
            "teasing_count": int(row["teasing_count"]),
            "samples": self._decode_samples(row["sample_messages"]),
            "pending_messages": int(row["pending_messages"])
            if row["pending_messages"] is not None
            else 0,
            "last_analyzed_at": int(row["last_analyzed_at"]) if row["last_analyzed_at"] is not None else None,
            "analysis_summary": row["analysis_summary"],
            "analysis_details": row["analysis_details"],
        }

    def increment_persona_pending(self, chat_id: int, user_id: int) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                SELECT pending_messages FROM persona_profiles
                WHERE chat_id = ? AND user_id = ?
                """,
                (chat_id, user_id),
            )
            row = cursor.fetchone()
            if row is None:
                pending = 1
                self._conn.execute(
                    """
                    INSERT INTO persona_profiles (
                        chat_id,
                        user_id,
                        summary,
                        details,
                        pending_messages,
                        last_analyzed_at
                    ) VALUES (?, ?, NULL, NULL, ?, NULL)
                    """,
                    (chat_id, user_id, pending),
                )
                return pending

            pending = int(row["pending_messages"]) + 1
            self._conn.execute(
                """
                UPDATE persona_profiles
                SET pending_messages = ?
                WHERE chat_id = ? AND user_id = ?
                """,
                (pending, chat_id, user_id),
            )
            return pending

    def get_persona_profile(
        self, chat_id: int, user_id: int
    ) -> Optional[dict[str, object]]:
        cursor = self._conn.execute(
            """
            SELECT summary, details, pending_messages, last_analyzed_at
            FROM persona_profiles
            WHERE chat_id = ? AND user_id = ?
            """,
            (chat_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "summary": row["summary"],
            "details": row["details"],
            "pending_messages": int(row["pending_messages"])
            if row["pending_messages"] is not None
            else 0,
            "last_analyzed_at": int(row["last_analyzed_at"])
            if row["last_analyzed_at"] is not None
            else None,
        }

    def save_persona_profile(
        self,
        chat_id: int,
        user_id: int,
        summary: str,
        details: str,
        analyzed_at: int,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO persona_profiles (
                    chat_id,
                    user_id,
                    summary,
                    details,
                    pending_messages,
                    last_analyzed_at
                ) VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(chat_id, user_id) DO UPDATE SET
                    summary = excluded.summary,
                    details = excluded.details,
                    pending_messages = 0,
                    last_analyzed_at = excluded.last_analyzed_at
                """,
                (chat_id, user_id, summary, details, analyzed_at),
            )


    def get_pair_messages(
        self, chat_id: int, speaker_id: int, target_id: int, limit: int
    ) -> List[sqlite3.Row]:
        cursor = self._conn.execute(
            """
            SELECT text, timestamp
            FROM pair_interaction_messages
            WHERE chat_id = ? AND speaker_id = ? AND target_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (chat_id, speaker_id, target_id, limit),
        )
        return cursor.fetchall()

    def get_chat_messages_before(
        self, chat_id: int, before_timestamp: int, limit: int
    ) -> List[sqlite3.Row]:
        cursor = self._conn.execute(
            """
            SELECT u.username, u.first_name, u.last_name, m.text, m.timestamp
            FROM messages m
            JOIN users u ON u.id = m.user_id
            WHERE m.chat_id = ? AND m.timestamp < ?
            ORDER BY m.timestamp DESC, m.id DESC
            LIMIT ?
            """,
            (chat_id, before_timestamp, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return rows

    def save_pair_analysis(
        self,
        chat_id: int,
        speaker_id: int,
        target_id: int,
        summary: str,
        details: str,
        analyzed_at: int,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE pair_interactions
                SET analysis_summary = ?,
                    analysis_details = ?,
                    last_analyzed_at = ?,
                    pending_messages = 0
                WHERE chat_id = ? AND speaker_id = ? AND target_id = ?
                """,
                (
                    summary,
                    details,
                    analyzed_at,
                    chat_id,
                    speaker_id,
                    target_id,
                ),
            )

    def get_user_by_id(self, user_id: int) -> Optional[sqlite3.Row]:
        cursor = self._conn.execute(
            "SELECT id, username, first_name, last_name FROM users WHERE id = ?",
            (user_id,),
        )
        return cursor.fetchone()

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
            WHERE m.chat_id = ? AND m.context_only = 0 {exclusion_clause}
            GROUP BY u.id
            ORDER BY message_count DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return cursor.fetchall()

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA)
            self._maybe_add_column(
                "messages",
                "context_only",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_interactions (
                    chat_id INTEGER NOT NULL,
                    speaker_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    total_count INTEGER NOT NULL DEFAULT 0,
                    informal_count INTEGER NOT NULL DEFAULT 0,
                    formal_count INTEGER NOT NULL DEFAULT 0,
                    teasing_count INTEGER NOT NULL DEFAULT 0,
                    sample_messages TEXT,
                    pending_messages INTEGER NOT NULL DEFAULT 0,
                    last_analyzed_at INTEGER,
                    analysis_summary TEXT,
                    analysis_details TEXT,
                    PRIMARY KEY (chat_id, speaker_id, target_id),
                    FOREIGN KEY(speaker_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_interaction_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    speaker_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    FOREIGN KEY(speaker_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pair_messages_lookup
                ON pair_interaction_messages(chat_id, speaker_id, target_id, timestamp DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS persona_profiles (
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    summary TEXT,
                    details TEXT,
                    pending_messages INTEGER NOT NULL DEFAULT 0,
                    last_analyzed_at INTEGER,
                    PRIMARY KEY (chat_id, user_id),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            self._maybe_add_column(
                "persona_profiles", "pending_messages", "INTEGER NOT NULL DEFAULT 0"
            )
            self._maybe_add_column(
                "persona_profiles", "last_analyzed_at", "INTEGER"
            )
            self._maybe_add_column(
                "pair_interactions", "pending_messages", "INTEGER NOT NULL DEFAULT 0"
            )
            self._maybe_add_column(
                "pair_interactions", "last_analyzed_at", "INTEGER"
            )
            self._maybe_add_column(
                "pair_interactions", "analysis_summary", "TEXT"
            )
            self._maybe_add_column(
                "pair_interactions", "analysis_details", "TEXT"
            )

    def _maybe_add_column(self, table: str, column: str, definition: str) -> None:
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        existing = {row["name"] for row in cursor.fetchall()}
        if column in existing:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _decode_samples(self, blob: Optional[str]) -> List[str]:
        if not blob:
            return []
        try:
            items = json.loads(blob)
        except json.JSONDecodeError:
            return []
        if not isinstance(items, list):
            return []
        result: List[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                normalized = item.strip()
                if normalized not in result:
                    result.append(normalized)
        return result

    def _encode_samples(self, samples: List[str]) -> str:
        return json.dumps(samples, ensure_ascii=False)

    def _normalize_sample(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        trimmed = text.strip()
        if not trimmed:
            return None
        if len(trimmed) > 280:
            trimmed = trimmed[:277].rstrip() + "..."
        return trimmed

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

    def reset_user_data(self, telegram_id: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "SELECT id FROM users WHERE telegram_id = ?",
                (telegram_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            user_id = int(row["id"])
            # purge personal messages
            self._conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            # purge pair interactions and their messages where user is speaker or target
            self._conn.execute(
                "DELETE FROM pair_interaction_messages WHERE speaker_id = ? OR target_id = ?",
                (user_id, user_id),
            )
            self._conn.execute(
                "DELETE FROM pair_interactions WHERE speaker_id = ? OR target_id = ?",
                (user_id, user_id),
            )
            # purge persona profile
            self._conn.execute(
                "DELETE FROM persona_profiles WHERE user_id = ?",
                (user_id,),
            )
            # purge any aliases for the user across chats
            self._conn.execute(
                "DELETE FROM user_aliases WHERE user_id = ?",
                (user_id,),
            )
            # purge persona preference if exists
            self._conn.execute(
                "DELETE FROM persona_preferences WHERE user_id = ?",
                (user_id,),
            )
            return True
