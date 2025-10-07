from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

telegram_stub = types.ModuleType("telegram")


class _StubTelegramMessage:
    pass


telegram_stub.Message = _StubTelegramMessage
sys.modules.setdefault("telegram", telegram_stub)

from bot_double.message_pipeline import MessagePipeline  # noqa: E402


class DummyUser:
    def __init__(
        self,
        telegram_id: int,
        *,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        is_bot: bool = False,
    ) -> None:
        self.id = telegram_id
        self.username = username
        self.first_name = first_name
        self.last_name = last_name
        self.is_bot = is_bot


class DummyMessage(_StubTelegramMessage):
    def __init__(
        self,
        text: str,
        user: DummyUser,
        *,
        chat_id: int,
        timestamp: int,
        reply_to: Optional["DummyMessage"] = None,
    ) -> None:
        self.text = text
        self.from_user = user
        self.chat_id = chat_id
        self.date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        self.via_bot = None
        self.forward_origin = None
        self.reply_to_message = reply_to


class DummyDB:
    def __init__(self) -> None:
        self.users: dict[int, int] = {}
        self.messages: List[Tuple[int, int, str, int, bool]] = []

    def upsert_user(
        self,
        telegram_id: int,
        username: Optional[str],
        first_name: Optional[str],
        last_name: Optional[str],
    ) -> int:
        if telegram_id not in self.users:
            self.users[telegram_id] = len(self.users) + 1
        return self.users[telegram_id]

    def store_message(
        self,
        chat_id: int,
        user_id: int,
        text: str,
        timestamp: int,
        *,
        context_only: bool = False,
    ) -> None:
        self.messages.append((chat_id, user_id, text, timestamp, context_only))


class DummySettings:
    min_tokens_to_store = 3
    enable_voice_transcription = False
    burst_gap_seconds = 30
    burst_max_duration_seconds = 120
    burst_inactivity_seconds = 10
    burst_max_parts = 6
    burst_max_chars = 2000
    burst_carryover_max_age_seconds = 120
    max_store_chars = 500
    enable_bursts = True


class DummyBot:
    def __init__(self) -> None:
        self._settings = DummySettings()
        self._db = DummyDB()
        self.db = self._db
        self._bot_id: Optional[int] = None
        self._bot_user_id: Optional[int] = None
        self._bot_name: Optional[str] = None
        self._bot_username: Optional[str] = None
        self.pair_calls: List[Tuple[DummyMessage, int, str]] = []
        self.persona_calls: List[Tuple[int, int, int]] = []
        self.force_direct_address: bool = False

    async def _run_db(self, func, *args, **kwargs):  # noqa: ANN001 - signature matches bot
        return func(*args, **kwargs)

    def _truncate_for_storage(self, text: str) -> str:
        limit = self._settings.max_store_chars
        if limit > 0:
            return text[:limit]
        return text

    async def _update_pair_interactions(
        self, message: DummyMessage, user_id: int, text: str
    ) -> None:
        self.pair_calls.append((message, user_id, text))

    async def _note_persona_message(
        self, chat_id: Optional[int], user_id: int, timestamp: int
    ) -> None:
        if chat_id is None:
            return
        self.persona_calls.append((chat_id, user_id, timestamp))

    async def _maybe_transcribe_voice_message(self, message: DummyMessage) -> Optional[str]:
        return None

    def _is_direct_address(self, message: DummyMessage, lowered_text: str) -> bool:
        return self.force_direct_address


class MessagePipelineCarryoverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = DummyBot()
        self.pipeline = MessagePipeline(self.bot)

    async def test_store_fragments_combines_carryover_to_meet_threshold(self) -> None:
        user = DummyUser(telegram_id=101)
        key = (42, 1)
        first = DummyMessage("ok", user, chat_id=42, timestamp=1)
        await self.pipeline._store_fragments(key, ["ok"], first, key[1], 1)
        self.assertIn(key, self.pipeline._carryover)

        second = DummyMessage("all good now", user, chat_id=42, timestamp=5)
        await self.pipeline._store_fragments(
            key, ["all good now"], second, key[1], 5
        )

        self.assertNotIn(key, self.pipeline._carryover)
        self.assertEqual(len(self.bot.db.messages), 1)
        chat_id, user_id, text, _, context_only = self.bot.db.messages[0]
        self.assertEqual(chat_id, 42)
        self.assertEqual(user_id, key[1])
        self.assertEqual(text, "ok all good now")
        self.assertFalse(context_only)
        self.assertEqual(len(self.bot.pair_calls), 1)
        self.assertIs(self.bot.pair_calls[0][0], second)
        self.assertEqual(len(self.bot.persona_calls), 1)
        self.assertEqual(self.bot.persona_calls[0][0], 42)

    async def test_flush_all_buffers_forces_carryover(self) -> None:
        user = DummyUser(telegram_id=202)
        key = (7, 3)
        message = DummyMessage("ok", user, chat_id=7, timestamp=2)
        await self.pipeline._store_fragments(key, ["ok"], message, key[1], 2)
        self.assertEqual(len(self.bot.db.messages), 0)

        await self.pipeline.flush_all_buffers()

        self.assertEqual(len(self.bot.db.messages), 1)
        stored = self.bot.db.messages[0]
        self.assertEqual(stored[2], "ok")
        self.assertEqual(len(self.bot.pair_calls), 1)
        self.assertIs(self.bot.pair_calls[0][0], message)
        self.assertEqual(len(self.bot.persona_calls), 1)
        self.assertEqual(self.bot.persona_calls[0][0], 7)

    async def test_expired_carryover_is_committed_before_new_fragments(self) -> None:
        user = DummyUser(telegram_id=303)
        key = (9, 4)
        first = DummyMessage("ok", user, chat_id=9, timestamp=0)
        await self.pipeline._store_fragments(key, ["ok"], first, key[1], 0)
        self.assertEqual(len(self.bot.db.messages), 0)

        self.bot._settings.burst_carryover_max_age_seconds = 1
        second = DummyMessage("fine today", user, chat_id=9, timestamp=5)
        await self.pipeline._store_fragments(
            key, ["fine today"], second, key[1], 5
        )

        self.assertEqual(len(self.bot.db.messages), 1)
        self.assertEqual(self.bot.db.messages[0][2], "ok")
        self.assertIn(key, self.pipeline._carryover)
        self.assertEqual(self.pipeline._carryover[key].texts, ["fine today"])
        self.assertEqual(len(self.bot.persona_calls), 1)
        self.assertEqual(self.bot.persona_calls[0][0], 9)

    async def test_direct_address_stored_as_context_only(self) -> None:
        user = DummyUser(telegram_id=404)
        message = DummyMessage(
            "бот привет как дела",
            user,
            chat_id=11,
            timestamp=10,
        )
        self.bot.force_direct_address = True

        result = await self.pipeline.capture_message(message)

        self.assertIsNotNone(result)
        self.assertEqual(len(self.bot.db.messages), 1)
        chat_id, user_id, text, _, context_only = self.bot.db.messages[0]
        self.assertEqual(chat_id, 11)
        self.assertEqual(user_id, self.bot.db.users[user.id])
        self.assertEqual(text, "бот привет как дела")
        self.assertTrue(context_only)
        self.assertEqual(len(self.bot.pair_calls), 0)
        self.assertEqual(len(self.bot.persona_calls), 0)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()
