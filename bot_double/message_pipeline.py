from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple, TYPE_CHECKING

from telegram import Message

from .burst_manager import BurstManager, BurstState
from .utils import (
    is_bufferable_message,
    should_store_context_snippet,
    should_store_message,
)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .bot import BotDouble


@dataclass
class ChatEvent:
    timestamp: int
    user_telegram_id: int
    reply_to_telegram_id: Optional[int]


class MessagePipeline:
    """Encapsulates message capture and persistence logic for BotDouble."""

    def __init__(self, bot: "BotDouble") -> None:
        self._bot = bot
        self._burst_manager: Optional[BurstManager] = None
        self._chat_events: Dict[int, Deque[ChatEvent]] = {}
        self._chat_events_maxlen = 50

    def attach_burst_manager(self, manager: BurstManager) -> None:
        self._burst_manager = manager

    async def capture_message(self, message: Message) -> Optional[Tuple[str, bool]]:
        if message.from_user is None:
            return None
        user = message.from_user
        bot_id = self._bot._bot_id  # accessing bot internals intentionally
        if user.is_bot:
            if bot_id is None or user.id != bot_id:
                return None
        if message.via_bot is not None:
            return None
        if message.forward_origin is not None:
            return None
        chat_id = message.chat_id
        if chat_id is None:
            return None
        user_id = await self._bot._run_db(
            self._bot._db.upsert_user,
            user.id,
            user.username,
            user.first_name,
            user.last_name,
        )
        if user.is_bot and user.id == bot_id:
            self._bot._bot_user_id = user_id
        timestamp = int(message.date.timestamp())
        key = (chat_id, user_id)

        text_source_voice = False
        text = message.text or ""
        if not text and self._bot._settings.enable_voice_transcription:
            text = await self._bot._maybe_transcribe_voice_message(message)
            if text:
                text_source_voice = True
        if not text:
            return None
        text = text.strip()
        if not text:
            return None

        if text_source_voice:
            if text.startswith(('/', '!', '.')):
                await self.flush_buffer_for_key(key)
                self._record_chat_event(message, timestamp)
                return None
            should_store = should_store_context_snippet(
                text, min_tokens=self._bot._settings.min_tokens_to_store
            )
            lowered = text.lower()
            bufferable = (
                not should_store
                and "http://" not in lowered
                and "https://" not in lowered
            )
        else:
            should_store = should_store_message(
                message,
                min_tokens=self._bot._settings.min_tokens_to_store,
                allowed_bot_id=bot_id,
            )
            bufferable = is_bufferable_message(
                message,
                allowed_bot_id=bot_id,
            )

        if should_store or bufferable:
            await self._append_to_burst(
                key,
                message,
                user_id,
                user.id,
                text,
                timestamp,
            )
        else:
            await self.flush_buffer_for_key(key)

        context_snippet = self._extract_command_context(text)
        if context_snippet and should_store_context_snippet(
            context_snippet, min_tokens=self._bot._settings.min_tokens_to_store
        ):
            context_snippet = self._bot._truncate_for_storage(context_snippet)
            await self._bot._run_db(
                self._bot._db.store_message,
                message.chat_id,
                user_id,
                context_snippet,
                timestamp,
                context_only=True,
            )

        self._record_chat_event(message, timestamp)
        return text, text_source_voice

    def should_break_burst(
        self,
        burst: BurstState,
        message: Message,
        user_telegram_id: int,
        timestamp: int,
    ) -> bool:
        settings = self._bot._settings
        if timestamp - burst.last_timestamp > settings.burst_gap_seconds:
            return True
        if settings.burst_max_duration_seconds > 0 and (
            timestamp - burst.start_timestamp
        ) > settings.burst_max_duration_seconds:
            return True
        reply_to_id = (
            message.reply_to_message.from_user.id
            if message.reply_to_message and message.reply_to_message.from_user
            else None
        )
        if self._was_interrupted_by_turn(
            burst.chat_id,
            burst.user_telegram_id,
            burst.last_timestamp,
            timestamp,
            reply_to_id,
        ):
            return True
        return False

    async def store_burst(self, burst: BurstState) -> None:
        combined_text = " ".join(burst.texts).strip()
        if not combined_text:
            return
        if not should_store_context_snippet(
            combined_text, min_tokens=self._bot._settings.min_tokens_to_store
        ):
            return
        combined_text = self._bot._truncate_for_storage(combined_text)
        await self._bot._run_db(
            self._bot._db.store_message,
            burst.chat_id,
            burst.user_id,
            combined_text,
            burst.last_timestamp,
            context_only=False,
        )
        await self._bot._update_pair_interactions(
            burst.last_message, burst.user_id, combined_text
        )
        await self._bot._note_persona_message(
            burst.chat_id, burst.user_id, burst.last_timestamp
        )

    async def flush_all_buffers(self) -> None:
        if self._burst_manager is None:
            return
        await self._burst_manager.flush_all()

    async def flush_buffers_for_chat(self, chat_id: int) -> None:
        if self._burst_manager is None:
            return
        await self._burst_manager.flush_for_chat(chat_id)

    async def flush_buffer_for_key(self, key: Tuple[int, int]) -> None:
        if self._burst_manager is None:
            return
        await self._burst_manager.flush_key(key)

    async def _append_to_burst(
        self,
        key: Tuple[int, int],
        message: Message,
        user_internal_id: int,
        user_telegram_id: int,
        text: str,
        timestamp: int,
    ) -> None:
        if self._burst_manager is None:
            return
        await self._burst_manager.append(
            key, message, user_internal_id, user_telegram_id, text, timestamp
        )

    def _extract_command_context(self, text: str) -> Optional[str]:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped.split(maxsplit=1)
        if not parts:
            return None
        command_token = parts[0]
        command = command_token.split("@", maxsplit=1)[0].lower()
        if command != "/imitate":
            return None
        if len(parts) < 2:
            return None
        remainder = parts[1].strip()
        return remainder or None

    def _was_interrupted_by_turn(
        self,
        chat_id: int,
        user_telegram_id: int,
        last_timestamp: int,
        current_timestamp: int,
        reply_to_telegram_id: Optional[int],
    ) -> bool:
        events = self._chat_events.get(chat_id)
        if not events or not self._bot._settings.enable_bursts:
            return False
        window = self._bot._settings.turn_window_seconds
        for event in events:
            if event.timestamp <= last_timestamp:
                continue
            if event.timestamp >= current_timestamp:
                break
            if event.user_telegram_id == user_telegram_id:
                continue
            if reply_to_telegram_id is not None and (
                event.user_telegram_id == reply_to_telegram_id
            ):
                return True
            if event.reply_to_telegram_id == user_telegram_id:
                return True
            if (
                current_timestamp - event.timestamp <= window
                and event.timestamp - last_timestamp <= window
            ):
                return True
        return False

    def _record_chat_event(self, message: Message, timestamp: int) -> None:
        chat_id = message.chat_id
        if chat_id is None or message.from_user is None:
            return
        reply_to_id = (
            message.reply_to_message.from_user.id
            if message.reply_to_message and message.reply_to_message.from_user
            else None
        )
        event = ChatEvent(
            timestamp=timestamp,
            user_telegram_id=message.from_user.id,
            reply_to_telegram_id=reply_to_id,
        )
        events = self._chat_events.get(chat_id)
        if events is None:
            events = deque(maxlen=self._chat_events_maxlen)
            self._chat_events[chat_id] = events
        events.append(event)
