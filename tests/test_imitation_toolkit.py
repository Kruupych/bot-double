from __future__ import annotations

import sys
import types
import unittest


telegram_stub = types.ModuleType("telegram")
telegram_stub.Message = object
telegram_stub.User = object
sys.modules.setdefault("telegram", telegram_stub)

from bot_double.imitation import ImitationToolkit


class ImitationToolkitStripCallSignsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.toolkit = ImitationToolkit(
            bot_name="Двойник",
            bot_username="bot_double",
            chain_cache_limit=1,
            answered_cache_limit=1,
        )

    def test_strip_call_signs_with_slash_delimiter(self) -> None:
        cleaned = self.toolkit.strip_call_signs("бот/двойник Вася привет")
        self.assertEqual(cleaned, "Вася привет")

    def test_strip_call_signs_with_alias_after_combo(self) -> None:
        cleaned = self.toolkit.strip_call_signs("бот/двойник @vasya привет")
        self.assertEqual(cleaned, "@vasya привет")

    def test_strip_call_signs_with_hyphenated_call(self) -> None:
        cleaned = self.toolkit.strip_call_signs("бот-двойник: расскажи шутку")
        self.assertEqual(cleaned, "расскажи шутку")

    def test_strip_call_signs_with_bot_prefix(self) -> None:
        cleaned = self.toolkit.strip_call_signs("бот Вася привет")
        self.assertEqual(cleaned, "Вася привет")

    def test_strip_call_signs_with_double_prefix(self) -> None:
        cleaned = self.toolkit.strip_call_signs("двойник Вася привет")
        self.assertEqual(cleaned, "Вася привет")

    def test_strip_call_signs_case_insensitive(self) -> None:
        cleaned = self.toolkit.strip_call_signs("Бот Вася привет")
        self.assertEqual(cleaned, "Вася привет")


if __name__ == "__main__":
    unittest.main()
