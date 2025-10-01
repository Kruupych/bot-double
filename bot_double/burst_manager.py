from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional, Tuple

from telegram import Message

LOGGER = logging.getLogger(__name__)


@dataclass
class BurstState:
    chat_id: int
    user_id: int
    user_telegram_id: int
    texts: list[str]
    start_timestamp: int
    last_timestamp: int
    last_message: Message
    total_chars: int
    task: Optional[asyncio.Task] = None


class BurstManager:
    """Aggregate short user messages into bursts before persisting them."""

    _Key = Tuple[int, int]

    def __init__(
        self,
        settings,
        *,
        should_break: Callable[[BurstState, Message, int, int], bool],
        flush_callback: Callable[[BurstState], Awaitable[None]],
    ) -> None:
        self._settings = settings
        self._should_break = should_break
        self._flush_callback = flush_callback
        self._states: Dict[BurstManager._Key, BurstState] = {}
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None

    async def append(
        self,
        key: _Key,
        message: Message,
        user_internal_id: int,
        user_telegram_id: int,
        text: str,
        timestamp: int,
    ) -> None:
        normalized = text.strip()
        if not normalized:
            await self.flush_key(key)
            return

        buffer_to_flush: Optional[BurstState] = None
        flush_now = False

        async with self._lock:
            burst = self._states.get(key)
            if burst:
                if self._should_break(burst, message, user_telegram_id, timestamp):
                    buffer_to_flush = self._states.pop(key)
                    if burst.task:
                        burst.task.cancel()
                    burst = None

            if burst is None:
                burst = BurstState(
                    chat_id=message.chat_id,
                    user_id=user_internal_id,
                    user_telegram_id=user_telegram_id,
                    texts=[normalized],
                    start_timestamp=timestamp,
                    last_timestamp=timestamp,
                    last_message=message,
                    total_chars=len(normalized),
                )
                self._states[key] = burst
            else:
                burst.texts.append(normalized)
                burst.last_timestamp = timestamp
                burst.last_message = message
                burst.total_chars += len(normalized)

            if burst.task:
                burst.task.cancel()
            burst.task = asyncio.create_task(self._delayed_flush(key))
            flush_now = self._burst_limits_exceeded(burst)

        if buffer_to_flush is not None:
            await self._flush_callback(buffer_to_flush)

        if flush_now:
            await self.flush_key(key)

    async def flush_key(self, key: _Key) -> None:
        burst = await self._pop_burst(key)
        if burst is None:
            return
        await self._flush_callback(burst)

    async def flush_all(self) -> None:
        async with self._lock:
            keys = list(self._states.keys())
        for key in keys:
            await self.flush_key(key)

    async def flush_for_chat(self, chat_id: int) -> None:
        to_flush: list[BurstManager._Key] = []
        async with self._lock:
            for key, burst in self._states.items():
                if burst.chat_id == chat_id:
                    to_flush.append(key)
        for key in to_flush:
            await self.flush_key(key)

    def start(self) -> None:
        if self._settings.enable_bursts and self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog())

    async def stop(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

    async def _delayed_flush(self, key: _Key) -> None:
        try:
            await asyncio.sleep(self._settings.burst_inactivity_seconds)
            await self.flush_key(key)
        except asyncio.CancelledError:  # cooperative cancel
            return

    async def _pop_burst(self, key: _Key) -> Optional[BurstState]:
        async with self._lock:
            burst = self._states.get(key)
            if not burst or not burst.texts:
                if burst and burst.task:
                    burst.task.cancel()
                    burst.task = None
                if burst:
                    del self._states[key]
                return None
            if burst.task:
                burst.task.cancel()
                burst.task = None
            del self._states[key]
            return burst

    def _burst_limits_exceeded(self, burst: BurstState) -> bool:
        if (
            self._settings.burst_max_parts > 0
            and len(burst.texts) >= self._settings.burst_max_parts
        ):
            return True
        if (
            self._settings.burst_max_chars > 0
            and burst.total_chars >= self._settings.burst_max_chars
        ):
            return True
        return False

    async def _watchdog(self) -> None:
        try:
            while True:
                await asyncio.sleep(5)
                if not self._settings.enable_bursts:
                    continue
                now = int(time.time())
                keys_to_flush: list[BurstManager._Key] = []
                inactivity = max(self._settings.burst_inactivity_seconds, 5)
                max_age = max(self._settings.burst_max_duration_seconds, 0)
                async with self._lock:
                    for key, burst in self._states.items():
                        if not burst.texts:
                            continue
                        if now - burst.last_timestamp >= inactivity * 2:
                            keys_to_flush.append(key)
                            continue
                        if (
                            max_age > 0
                            and (now - burst.start_timestamp) > max_age + inactivity
                        ):
                            keys_to_flush.append(key)
                for key in keys_to_flush:
                    await self.flush_key(key)
        except asyncio.CancelledError:
            LOGGER.debug("Burst watchdog cancelled")
            return

