from __future__ import annotations

import asyncio
import sys
import types
import unittest
from typing import List, Set

# Stub telegram module before importing bot_double modules
telegram_stub = types.ModuleType("telegram")
telegram_stub.Message = object
telegram_stub.User = object
sys.modules.setdefault("telegram", telegram_stub)

from bot_double.analysis_worker import AnalysisWorker


class MockWorker(AnalysisWorker[int]):
    """Concrete implementation of AnalysisWorker for testing."""

    def __init__(self) -> None:
        super().__init__(name="Testable")
        self.processed_keys: List[int] = []
        self.should_queue_result: bool = True
        self.raise_on_analysis: bool = False

    async def _should_queue(self, key: int) -> bool:
        return self.should_queue_result

    async def _perform_analysis(self, key: int) -> None:
        if self.raise_on_analysis:
            raise ValueError("Test error")
        self.processed_keys.append(key)


class AnalysisWorkerBasicTests(unittest.IsolatedAsyncioTestCase):
    """Basic tests for AnalysisWorker."""

    async def test_initial_state(self) -> None:
        """Test worker initial state."""
        worker = MockWorker()
        self.assertFalse(worker.is_running)
        self.assertIsNone(worker._queue)
        self.assertIsNone(worker._worker_task)
        self.assertEqual(len(worker._inflight), 0)

    async def test_start_creates_queue_and_task(self) -> None:
        """Test that start() creates queue and task."""
        worker = MockWorker()
        worker.start()

        self.assertTrue(worker.is_running)
        self.assertIsNotNone(worker._queue)
        self.assertIsNotNone(worker._worker_task)

        await worker.stop()

    async def test_stop_cleans_up(self) -> None:
        """Test that stop() cleans up properly."""
        worker = MockWorker()
        worker.start()
        await worker.stop()

        self.assertFalse(worker.is_running)
        self.assertIsNone(worker._queue)
        self.assertIsNone(worker._worker_task)
        self.assertEqual(len(worker._inflight), 0)

    async def test_double_start_is_safe(self) -> None:
        """Test that calling start() twice is safe."""
        worker = MockWorker()
        worker.start()
        first_task = worker._worker_task

        worker.start()
        second_task = worker._worker_task

        # Should be the same task
        self.assertIs(first_task, second_task)

        await worker.stop()

    async def test_double_stop_is_safe(self) -> None:
        """Test that calling stop() twice is safe."""
        worker = MockWorker()
        worker.start()
        await worker.stop()
        await worker.stop()  # Should not raise


class AnalysisWorkerQueueTests(unittest.IsolatedAsyncioTestCase):
    """Tests for AnalysisWorker queue behavior."""

    async def test_maybe_queue_when_not_running(self) -> None:
        """Test that maybe_queue does nothing when worker not running."""
        worker = MockWorker()
        await worker.maybe_queue(1)
        # Should not raise, just do nothing

    async def test_maybe_queue_adds_to_queue(self) -> None:
        """Test that maybe_queue adds item to queue."""
        worker = MockWorker()
        worker.start()

        await worker.maybe_queue(42)

        # Wait for processing
        await asyncio.sleep(0.1)

        self.assertIn(42, worker.processed_keys)

        await worker.stop()

    async def test_maybe_queue_respects_should_queue(self) -> None:
        """Test that maybe_queue respects _should_queue result."""
        worker = MockWorker()
        worker.should_queue_result = False
        worker.start()

        await worker.maybe_queue(42)

        # Give time to process (if it were queued)
        await asyncio.sleep(0.1)

        # Should not be processed
        self.assertNotIn(42, worker.processed_keys)

        await worker.stop()

    async def test_duplicate_keys_not_queued(self) -> None:
        """Test that duplicate keys are not added while in flight."""
        worker = MockWorker()
        worker.start()

        # Add to inflight manually to simulate in-progress item
        worker._inflight.add(42)

        # Try to queue same key
        await worker.maybe_queue(42)

        # Should not be in queue again
        self.assertEqual(worker._queue.qsize(), 0)

        await worker.stop()

    async def test_multiple_items_processed_in_order(self) -> None:
        """Test that multiple items are processed in order."""
        worker = MockWorker()
        worker.start()

        await worker.maybe_queue(1)
        await worker.maybe_queue(2)
        await worker.maybe_queue(3)

        # Wait for processing
        await asyncio.sleep(0.2)

        self.assertEqual(worker.processed_keys, [1, 2, 3])

        await worker.stop()


class AnalysisWorkerErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    """Tests for AnalysisWorker error handling."""

    async def test_analysis_error_does_not_stop_worker(self) -> None:
        """Test that error in analysis does not stop the worker."""
        worker = MockWorker()
        worker.raise_on_analysis = True
        worker.start()

        await worker.maybe_queue(1)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Worker should still be running
        self.assertTrue(worker.is_running)

        # Key should be removed from inflight even on error
        self.assertNotIn(1, worker._inflight)

        await worker.stop()

    async def test_analysis_error_allows_next_item(self) -> None:
        """Test that worker continues processing after error."""
        worker = MockWorker()
        worker.start()

        # First item will raise
        worker.raise_on_analysis = True
        await worker.maybe_queue(1)
        await asyncio.sleep(0.1)

        # Second item should succeed
        worker.raise_on_analysis = False
        await worker.maybe_queue(2)
        await asyncio.sleep(0.1)

        self.assertNotIn(1, worker.processed_keys)  # Failed
        self.assertIn(2, worker.processed_keys)  # Succeeded

        await worker.stop()


class AnalysisWorkerInflightTests(unittest.IsolatedAsyncioTestCase):
    """Tests for AnalysisWorker inflight tracking."""

    async def test_inflight_cleared_after_processing(self) -> None:
        """Test that key is removed from inflight after processing."""
        worker = MockWorker()
        worker.start()

        await worker.maybe_queue(42)
        await asyncio.sleep(0.1)

        # Key should be processed and removed from inflight
        self.assertIn(42, worker.processed_keys)
        self.assertNotIn(42, worker._inflight)

        await worker.stop()

    async def test_inflight_cleared_on_stop(self) -> None:
        """Test that inflight is cleared when worker stops."""
        worker = MockWorker()
        worker.start()

        # Add items to inflight
        worker._inflight.add(1)
        worker._inflight.add(2)

        await worker.stop()

        self.assertEqual(len(worker._inflight), 0)


class TupleKeyWorker(AnalysisWorker[tuple]):
    """Worker with tuple keys for testing generic behavior."""

    def __init__(self) -> None:
        super().__init__(name="TupleKey")
        self.processed_keys: List[tuple] = []

    async def _should_queue(self, key: tuple) -> bool:
        return True

    async def _perform_analysis(self, key: tuple) -> None:
        self.processed_keys.append(key)


class AnalysisWorkerGenericKeyTests(unittest.IsolatedAsyncioTestCase):
    """Tests for AnalysisWorker with different key types."""

    async def test_tuple_keys(self) -> None:
        """Test worker with tuple keys (like relationship worker)."""
        worker = TupleKeyWorker()
        worker.start()

        key = (1, 2, 3)
        await worker.maybe_queue(key)
        await asyncio.sleep(0.1)

        self.assertIn(key, worker.processed_keys)

        await worker.stop()

    async def test_tuple_keys_deduplication(self) -> None:
        """Test that tuple keys are properly deduplicated."""
        worker = TupleKeyWorker()
        worker.start()

        key = (1, 2, 3)
        worker._inflight.add(key)

        # Same tuple should not be queued
        await worker.maybe_queue(key)

        self.assertEqual(worker._queue.qsize(), 0)

        await worker.stop()


if __name__ == "__main__":
    unittest.main()

