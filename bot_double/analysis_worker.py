from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Generic, Optional, Set, TypeVar

LOGGER = logging.getLogger(__name__)

K = TypeVar("K")  # Key type for queue items


class AnalysisWorker(ABC, Generic[K]):
    """
    Base class for background analysis workers.
    
    Subclasses must implement:
    - _perform_analysis: The actual analysis work for a key
    - _should_queue: Check if a key should be queued for analysis
    """

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._queue: Optional[asyncio.Queue[K]] = None
        self._inflight: Set[K] = set()
        self._worker_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None

    def start(self) -> None:
        """Start the worker. Must be called from an async context."""
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Stop the worker and clean up."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
            self._queue = None
        self._inflight.clear()

    async def maybe_queue(self, key: K) -> None:
        """Queue a key for analysis if conditions are met."""
        if self._queue is None:
            return
        if key in self._inflight:
            return
        if not await self._should_queue(key):
            return
        await self._queue.put(key)
        self._inflight.add(key)

    @abstractmethod
    async def _should_queue(self, key: K) -> bool:
        """Check if a key should be queued for analysis."""
        ...

    @abstractmethod
    async def _perform_analysis(self, key: K) -> None:
        """Perform the analysis for a key."""
        ...

    async def _worker_loop(self) -> None:
        """Main worker loop that processes queued items."""
        assert self._queue is not None
        try:
            while True:
                key = await self._queue.get()
                try:
                    await self._perform_analysis(key)
                except Exception:
                    LOGGER.exception("%s analysis task failed", self._name)
                finally:
                    self._inflight.discard(key)
                    self._queue.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            if self._queue is not None:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except asyncio.QueueEmpty:
                        break

