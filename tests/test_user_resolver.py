from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stub telegram module before importing bot_double modules
telegram_stub = types.ModuleType("telegram")
telegram_stub.Message = object
telegram_stub.User = object
sys.modules.setdefault("telegram", telegram_stub)

from bot_double.db import Database
from bot_double.user_resolver import (
    UserResolverService,
    SCORE_EXACT_MATCH,
    SCORE_SUBSTRING_MATCH,
    SCORE_MINIMUM_THRESHOLD,
)


class UserResolverServiceTests(unittest.IsolatedAsyncioTestCase):
    """Tests for UserResolverService."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.db = Database(Path(self._tmp.name), max_messages_per_user=100)

        async def run_db(func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        self.resolver = UserResolverService(
            db=self.db,
            run_db=run_db,
            get_bot_user_id=lambda: None,
        )

    async def asyncTearDown(self) -> None:
        self.db.close()
        os.unlink(self._tmp.name)

    # --- Tests for resolve_user_descriptor ---

    async def test_resolve_by_username_with_at_sign(self) -> None:
        """Test resolving user by @username."""
        user_id = self.db.upsert_user(111, "testuser", "Test", "User")
        row, suggestions = await self.resolver.resolve_user_descriptor(None, "@testuser")
        self.assertIsNotNone(row)
        self.assertEqual(int(row["id"]), user_id)
        self.assertEqual(suggestions, [])

    async def test_resolve_unknown_username_returns_none(self) -> None:
        """Test that unknown username returns None."""
        row, suggestions = await self.resolver.resolve_user_descriptor(None, "@unknown")
        self.assertIsNone(row)
        self.assertEqual(suggestions, [])

    async def test_resolve_empty_descriptor(self) -> None:
        """Test that empty descriptor returns None."""
        row, suggestions = await self.resolver.resolve_user_descriptor(1, "")
        self.assertIsNone(row)
        self.assertEqual(suggestions, [])

    async def test_resolve_whitespace_descriptor(self) -> None:
        """Test that whitespace-only descriptor returns None."""
        row, suggestions = await self.resolver.resolve_user_descriptor(1, "   ")
        self.assertIsNone(row)
        self.assertEqual(suggestions, [])

    async def test_resolve_without_chat_id_for_non_username(self) -> None:
        """Test that non-username lookup without chat_id returns None."""
        self.db.upsert_user(111, "testuser", "Test", "User")
        row, suggestions = await self.resolver.resolve_user_descriptor(None, "Test")
        self.assertIsNone(row)
        self.assertEqual(suggestions, [])

    async def test_resolve_by_alias(self) -> None:
        """Test resolving user by alias."""
        chat_id = 42
        user_id = self.db.upsert_user(111, "testuser", "Test", "User")
        self.db.add_aliases(chat_id, user_id, ["тестовый"])

        row, suggestions = await self.resolver.resolve_user_descriptor(chat_id, "тестовый")
        self.assertIsNotNone(row)
        self.assertEqual(int(row["id"]), user_id)

    async def test_resolve_by_first_name(self) -> None:
        """Test resolving user by first name in chat participants."""
        chat_id = 42
        user_id = self.db.upsert_user(111, "testuser", "Петя", None)
        # Store a message to make user a chat participant
        self.db.store_message(chat_id, user_id, "тестовое сообщение", 1000)

        row, suggestions = await self.resolver.resolve_user_descriptor(chat_id, "Петя")
        self.assertIsNotNone(row)
        self.assertEqual(int(row["id"]), user_id)

    async def test_resolve_by_transliterated_name(self) -> None:
        """Test resolving user by transliterated name."""
        chat_id = 42
        user_id = self.db.upsert_user(111, "testuser", "Вася", None)
        self.db.store_message(chat_id, user_id, "тестовое сообщение", 1000)

        # Should match "vasya" -> "вася"
        row, suggestions = await self.resolver.resolve_user_descriptor(chat_id, "vasya")
        self.assertIsNotNone(row)
        self.assertEqual(int(row["id"]), user_id)

    async def test_resolve_multiple_candidates_returns_suggestions(self) -> None:
        """Test that multiple similar candidates return suggestions."""
        chat_id = 42
        user_id1 = self.db.upsert_user(111, "user1", "Саша", "Иванов")
        user_id2 = self.db.upsert_user(222, "user2", "Саша", "Петров")
        self.db.store_message(chat_id, user_id1, "сообщение один", 1000)
        self.db.store_message(chat_id, user_id2, "сообщение два", 1001)

        row, suggestions = await self.resolver.resolve_user_descriptor(chat_id, "Саша")
        # Should return suggestions, not a single match
        self.assertIsNone(row)
        self.assertGreater(len(suggestions), 0)

    # --- Tests for cache operations ---

    async def test_cache_invalidation(self) -> None:
        """Test that cache invalidation works."""
        chat_id = 42
        # Populate cache
        await self.resolver.get_alias_maps(chat_id)
        self.assertIn(chat_id, self.resolver._alias_cache)

        # Invalidate
        self.resolver.invalidate_cache(chat_id)
        self.assertNotIn(chat_id, self.resolver._alias_cache)
        self.assertNotIn(chat_id, self.resolver._alias_display_cache)

    async def test_cache_invalidation_with_none(self) -> None:
        """Test that cache invalidation with None does nothing."""
        chat_id = 42
        await self.resolver.get_alias_maps(chat_id)
        self.assertIn(chat_id, self.resolver._alias_cache)

        self.resolver.invalidate_cache(None)
        # Should still be cached
        self.assertIn(chat_id, self.resolver._alias_cache)

    async def test_clear_all_caches(self) -> None:
        """Test clearing all caches."""
        # Populate caches for multiple chats
        await self.resolver.get_alias_maps(1)
        await self.resolver.get_alias_maps(2)

        self.resolver.clear_all_caches()
        self.assertEqual(len(self.resolver._alias_cache), 0)
        self.assertEqual(len(self.resolver._alias_display_cache), 0)

    async def test_alias_cache_is_used(self) -> None:
        """Test that alias cache is actually used on second call."""
        chat_id = 42
        user_id = self.db.upsert_user(111, "testuser", "Test", None)
        self.db.add_aliases(chat_id, user_id, ["алиас"])

        # First call - populates cache
        map1, display1 = await self.resolver.get_alias_maps(chat_id)
        self.assertIn("алиас", map1)

        # Modify DB directly (cache should still return old data)
        self.db.add_aliases(chat_id, user_id, ["новыйалиас"])

        # Second call - should return cached data
        map2, display2 = await self.resolver.get_alias_maps(chat_id)
        self.assertNotIn("новыйалиас", map2)

        # After invalidation, new data should be returned
        self.resolver.invalidate_cache(chat_id)
        map3, display3 = await self.resolver.get_alias_maps(chat_id)
        self.assertIn("новыйалиас", map3)

    # --- Tests for _generate_variants ---

    def test_generate_variants_basic(self) -> None:
        """Test generating variants from values."""
        variants = self.resolver._generate_variants(["Петя", "Вася"])
        self.assertIn("петя", variants)
        self.assertIn("petya", variants)
        self.assertIn("вася", variants)
        self.assertIn("vasya", variants)

    def test_generate_variants_empty_values(self) -> None:
        """Test generating variants from empty input."""
        variants = self.resolver._generate_variants([])
        self.assertEqual(len(variants), 0)

    def test_generate_variants_skips_empty_strings(self) -> None:
        """Test that empty strings are skipped."""
        variants = self.resolver._generate_variants(["", "   ", "тест"])
        self.assertIn("тест", variants)
        self.assertNotIn("", variants)

    # --- Tests for _descriptor_variants ---

    def test_descriptor_variants_basic(self) -> None:
        """Test basic descriptor variant generation."""
        variants = self.resolver._descriptor_variants("Тест")
        self.assertIn("тест", variants)
        self.assertIn("test", variants)

    def test_descriptor_variants_with_quotes(self) -> None:
        """Test descriptor variants strips quotes."""
        variants = self.resolver._descriptor_variants('"Тест"')
        self.assertIn("тест", variants)

    def test_descriptor_variants_with_delimiter(self) -> None:
        """Test descriptor variants handles delimiters."""
        variants = self.resolver._descriptor_variants("Имя: описание")
        self.assertIn("имя", variants)

    def test_descriptor_variants_with_multiple_tokens(self) -> None:
        """Test descriptor variants handles multiple tokens."""
        variants = self.resolver._descriptor_variants("Иван Петров")
        self.assertIn("иван петров", variants)
        self.assertIn("иван", variants)

    # --- Tests for _score_descriptor ---

    def test_score_descriptor_exact_match(self) -> None:
        """Test exact match returns 1.0."""
        score = self.resolver._score_descriptor(["тест"], ["тест"])
        self.assertEqual(score, SCORE_EXACT_MATCH)

    def test_score_descriptor_substring_match(self) -> None:
        """Test substring match returns high score."""
        score = self.resolver._score_descriptor(["тест"], ["тестовый"])
        self.assertGreaterEqual(score, SCORE_SUBSTRING_MATCH)

    def test_score_descriptor_no_match(self) -> None:
        """Test no match returns low score."""
        score = self.resolver._score_descriptor(["abc"], ["xyz"])
        self.assertLess(score, SCORE_MINIMUM_THRESHOLD)

    def test_score_descriptor_empty_candidates(self) -> None:
        """Test empty candidates returns 0."""
        score = self.resolver._score_descriptor(["тест"], [])
        self.assertEqual(score, 0.0)

    def test_score_descriptor_token_overlap(self) -> None:
        """Test token overlap scoring."""
        score = self.resolver._score_descriptor(["иван петров"], ["иван сидоров"])
        self.assertGreater(score, 0.5)


class UserResolverBotExclusionTests(unittest.IsolatedAsyncioTestCase):
    """Tests for bot user exclusion."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.db = Database(Path(self._tmp.name), max_messages_per_user=100)
        self.bot_user_id: Optional[int] = None

        async def run_db(func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        self.resolver = UserResolverService(
            db=self.db,
            run_db=run_db,
            get_bot_user_id=lambda: self.bot_user_id,
        )

    async def asyncTearDown(self) -> None:
        self.db.close()
        os.unlink(self._tmp.name)

    async def test_bot_user_excluded_from_resolution(self) -> None:
        """Test that bot user is excluded from fuzzy resolution."""
        chat_id = 42
        bot_internal_id = self.db.upsert_user(999, "bot_user", "Bot", None)
        user_id = self.db.upsert_user(111, "real_user", "Бот", None)  # Similar name

        self.db.store_message(chat_id, bot_internal_id, "сообщение бота", 1000)
        self.db.store_message(chat_id, user_id, "сообщение юзера", 1001)

        # Set bot user ID
        self.bot_user_id = bot_internal_id

        # Should resolve to real user, not bot
        row, suggestions = await self.resolver.resolve_user_descriptor(chat_id, "Бот")
        if row is not None:
            self.assertEqual(int(row["id"]), user_id)


if __name__ == "__main__":
    unittest.main()

