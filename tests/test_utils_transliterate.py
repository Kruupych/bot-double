from __future__ import annotations

import sys
import types
import unittest

# Stub telegram module before importing bot_double modules
telegram_stub = types.ModuleType("telegram")
telegram_stub.Message = object
telegram_stub.User = object
sys.modules.setdefault("telegram", telegram_stub)

from bot_double.utils import transliterate, normalize_alias


class TransliterateTests(unittest.TestCase):
    """Tests for the transliterate function."""

    def test_basic_cyrillic(self) -> None:
        self.assertEqual(transliterate("привет"), "privet")

    def test_full_alphabet(self) -> None:
        # Test all Cyrillic characters
        self.assertEqual(transliterate("а"), "a")
        self.assertEqual(transliterate("б"), "b")
        self.assertEqual(transliterate("в"), "v")
        self.assertEqual(transliterate("г"), "g")
        self.assertEqual(transliterate("д"), "d")
        self.assertEqual(transliterate("е"), "e")
        self.assertEqual(transliterate("ё"), "e")
        self.assertEqual(transliterate("ж"), "zh")
        self.assertEqual(transliterate("з"), "z")
        self.assertEqual(transliterate("и"), "i")
        self.assertEqual(transliterate("й"), "y")
        self.assertEqual(transliterate("к"), "k")
        self.assertEqual(transliterate("л"), "l")
        self.assertEqual(transliterate("м"), "m")
        self.assertEqual(transliterate("н"), "n")
        self.assertEqual(transliterate("о"), "o")
        self.assertEqual(transliterate("п"), "p")
        self.assertEqual(transliterate("р"), "r")
        self.assertEqual(transliterate("с"), "s")
        self.assertEqual(transliterate("т"), "t")
        self.assertEqual(transliterate("у"), "u")
        self.assertEqual(transliterate("ф"), "f")
        self.assertEqual(transliterate("х"), "h")
        self.assertEqual(transliterate("ц"), "ts")
        self.assertEqual(transliterate("ч"), "ch")
        self.assertEqual(transliterate("ш"), "sh")
        self.assertEqual(transliterate("щ"), "shch")
        self.assertEqual(transliterate("ъ"), "")
        self.assertEqual(transliterate("ы"), "y")
        self.assertEqual(transliterate("ь"), "")
        self.assertEqual(transliterate("э"), "e")
        self.assertEqual(transliterate("ю"), "yu")
        self.assertEqual(transliterate("я"), "ya")

    def test_mixed_text(self) -> None:
        self.assertEqual(transliterate("hello мир"), "hello mir")
        self.assertEqual(transliterate("test123тест"), "test123test")

    def test_empty_string(self) -> None:
        self.assertEqual(transliterate(""), "")

    def test_latin_only(self) -> None:
        self.assertEqual(transliterate("hello123"), "hello123")
        self.assertEqual(transliterate("ABC"), "ABC")

    def test_special_characters(self) -> None:
        self.assertEqual(transliterate("щука"), "shchuka")
        self.assertEqual(transliterate("ёжик"), "ezhik")
        self.assertEqual(transliterate("юля"), "yulya")

    def test_names(self) -> None:
        self.assertEqual(transliterate("петя"), "petya")
        self.assertEqual(transliterate("вася"), "vasya")
        self.assertEqual(transliterate("маша"), "masha")
        self.assertEqual(transliterate("саша"), "sasha")

    def test_with_spaces_and_punctuation(self) -> None:
        self.assertEqual(transliterate("привет мир!"), "privet mir!")
        self.assertEqual(transliterate("как дела?"), "kak dela?")

    def test_soft_and_hard_signs_removed(self) -> None:
        self.assertEqual(transliterate("объект"), "obekt")
        self.assertEqual(transliterate("большой"), "bolshoy")


class NormalizeAliasTests(unittest.TestCase):
    """Tests for the normalize_alias function."""

    def test_basic_normalization(self) -> None:
        self.assertEqual(normalize_alias("Тест"), "тест")
        self.assertEqual(normalize_alias("  Test  "), "test")

    def test_replaces_underscores_and_hyphens(self) -> None:
        self.assertEqual(normalize_alias("some_name"), "some name")
        self.assertEqual(normalize_alias("some-name"), "some name")

    def test_removes_special_characters(self) -> None:
        self.assertEqual(normalize_alias("test!@#"), "test")
        self.assertEqual(normalize_alias("@username"), "username")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_alias("  multiple   spaces  "), "multiple spaces")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_alias(""), "")
        self.assertEqual(normalize_alias("   "), "")


if __name__ == "__main__":
    unittest.main()

