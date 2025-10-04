from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from telegram import Message

from .burst_manager import BurstManager, BurstState
from .utils import (
    is_bufferable_message,
    should_store_context_snippet,
    should_store_message,
)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .bot import BotDouble
@dataclass(slots=True)
class CarryoverBuffer:
    texts: List[str]
    last_message: Message
    last_timestamp: int
    user_id: int


class MessagePipeline:
    """Encapsulates message capture and persistence logic for BotDouble."""

    def __init__(self, bot: "BotDouble") -> None:
        self._bot = bot
        self._burst_manager: Optional[BurstManager] = None
        self._carryover: Dict[Tuple[int, int], CarryoverBuffer] = {}

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

        if should_store:
            await self.flush_buffer_for_key(key)
            await self._store_fragments(
                key,
                [text],
                message,
                user_id,
                timestamp,
            )
        elif bufferable:
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

        return text, text_source_voice

    def should_break_burst(
        self,
        burst: BurstState,
        message: Message,
        user_telegram_id: int,
        timestamp: int,
    ) -> bool:
        settings = self._bot._settings
        if user_telegram_id != burst.user_telegram_id:
            return True
        if timestamp - burst.last_timestamp > settings.burst_gap_seconds:
            return True
        if settings.burst_max_duration_seconds > 0 and (
            timestamp - burst.start_timestamp
        ) > settings.burst_max_duration_seconds:
            return True
        return False

    async def store_burst(self, burst: BurstState) -> None:
        await self._store_fragments(
            (burst.chat_id, burst.user_id),
            burst.texts,
            burst.last_message,
            burst.user_id,
            burst.last_timestamp,
        )

    async def flush_all_buffers(self) -> None:
        if self._burst_manager is not None:
            await self._burst_manager.flush_all()
        await self._flush_all_carryover(force=True)

    async def flush_buffers_for_chat(self, chat_id: int) -> None:
        if self._burst_manager is not None:
            await self._burst_manager.flush_for_chat(chat_id)
        await self._flush_carryover_for_chat(chat_id, force=True)

    async def flush_buffer_for_key(self, key: Tuple[int, int]) -> None:
        if self._burst_manager is not None:
            await self._burst_manager.flush_key(key)
        await self._flush_carryover_for_key(key, force=False)

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

    async def _store_fragments(
        self,
        key: Tuple[int, int],
        fragments: List[str],
        message: Message,
        user_id: int,
        timestamp: int,
        *,
        force: bool = False,
    ) -> None:
        if not fragments:
            return
        carryover = self._carryover.pop(key, None)
        pieces: List[str] = []
        if carryover and carryover.texts:
            if self._carryover_is_expired(carryover, timestamp):
                await self._persist_text(
                    carryover.texts,
                    carryover.last_message,
                    carryover.user_id,
                    carryover.last_timestamp,
                    force=True,
                )
            else:
                pieces.extend(carryover.texts)
        pieces.extend(fragments)
        stored = await self._persist_text(
            pieces,
            message,
            user_id,
            timestamp,
            force=force,
        )
        if not stored:
            self._carryover[key] = CarryoverBuffer(
                texts=pieces,
                last_message=message,
                last_timestamp=timestamp,
                user_id=user_id,
            )

    async def _persist_text(
        self,
        fragments: List[str],
        message: Message,
        user_id: int,
        timestamp: int,
        *,
        force: bool,
    ) -> bool:
        combined_text = " ".join(fragments).strip()
        if not combined_text:
            return True
        meets_threshold = should_store_context_snippet(
            combined_text, min_tokens=self._bot._settings.min_tokens_to_store
        )
        if not meets_threshold and not force:
            return False
        chat_id = message.chat_id
        if chat_id is None:
            return True
        combined_text = self._bot._truncate_for_storage(combined_text)
        await self._bot._run_db(
            self._bot._db.store_message,
            chat_id,
            user_id,
            combined_text,
            timestamp,
            context_only=False,
        )
        await self._bot._update_pair_interactions(message, user_id, combined_text)
        await self._bot._note_persona_message(chat_id, user_id, timestamp)
        return True

    def _carryover_is_expired(
        self, carryover: CarryoverBuffer, current_timestamp: int
    ) -> bool:
        max_age = self._bot._settings.burst_carryover_max_age_seconds
        if max_age <= 0:
            return False
        return (current_timestamp - carryover.last_timestamp) > max_age

    async def _flush_carryover_for_key(
        self, key: Tuple[int, int], *, force: bool
    ) -> None:
        buffer = self._carryover.pop(key, None)
        if buffer is None:
            return
        stored = await self._persist_text(
            buffer.texts,
            buffer.last_message,
            buffer.user_id,
            buffer.last_timestamp,
            force=force,
        )
        if not stored:
            self._carryover[key] = buffer

    async def _flush_carryover_for_chat(self, chat_id: int, *, force: bool) -> None:
        keys = [key for key in self._carryover.keys() if key[0] == chat_id]
        for key in keys:
            await self._flush_carryover_for_key(key, force=force)

    async def _flush_all_carryover(self, *, force: bool) -> None:
        keys = list(self._carryover.keys())
        for key in keys:
            await self._flush_carryover_for_key(key, force=force)
