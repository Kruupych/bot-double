from __future__ import annotations

import re
import unicodedata
from typing import Optional

from telegram import Message


def display_name(username: Optional[str], first_name: Optional[str], last_name: Optional[str]) -> str:
    parts = [part for part in (first_name, last_name) if part]
    if parts:
        return " ".join(parts)
    if username:
        return f"@{username}" if not username.startswith("@") else username
    return "Неизвестный пользователь"


def should_store_message(
    message: Message, *, min_tokens: int, allowed_bot_id: Optional[int] = None
) -> bool:
    if message.from_user and message.from_user.is_bot:
        if allowed_bot_id is None or message.from_user.id != allowed_bot_id:
            return False
    if not message.text:
        return False
    if message.text.startswith("/"):
        return False
    if message.text.startswith("!"):
        return False
    if message.text.startswith("."):
        return False
    if message.via_bot is not None:
        return False
    if message.forward_origin is not None:
        return False
    return _passes_text_filters(message.text, min_tokens=min_tokens)


def is_bufferable_message(
    message: Message, *, allowed_bot_id: Optional[int] = None
) -> bool:
    if message.from_user and message.from_user.is_bot:
        if allowed_bot_id is None or message.from_user.id != allowed_bot_id:
            return False
    text = message.text or ""
    if not text:
        return False
    if text.startswith("/"):
        return False
    if text.startswith("!"):
        return False
    if text.startswith("."):
        return False
    if message.via_bot is not None:
        return False
    if message.forward_origin is not None:
        return False
    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered:
        return False
    return True


def _passes_text_filters(text: str, *, min_tokens: int) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    tokens = stripped.replace("\n", " ").split()
    if len(tokens) < min_tokens:
        return False
    lowered = stripped.lower()
    if "http://" in lowered or "https://" in lowered:
        return False
    return True


def should_store_context_snippet(text: str, *, min_tokens: int) -> bool:
    return _passes_text_filters(text, min_tokens=min_tokens)


_ALIAS_SANITIZE_RE = re.compile(r"[^0-9a-zа-яё\s]")


def normalize_alias(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = _ALIAS_SANITIZE_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


_FEMALE_ENDINGS = ("а", "я", "ия", "ля", "ся", "на", "ра")
_MALE_EXCEPTIONS = {
    "никита",
    "илья",
    "кузьма",
    "миша",
    "жека",
    "шура",
    "женя",
    "саша",
    "валя",
}
_FEMALE_OVERRIDES = {
    "женя",
    "саша",
    "валя",
    "ксюша",
    "лера",
    "аня",
    "оля",
    "лена",
    "инна",
    "мария",
    "марина",
    "елена",
    "алёна",
    "настя",
    "катя",
    "инга",
    "диана",
    "лара",
}


def guess_gender(first_name: Optional[str], username: Optional[str]) -> Optional[str]:
    if not first_name:
        return None
    name = first_name.strip().lower()
    if not name:
        return None
    if name in _FEMALE_OVERRIDES:
        return "female"
    if name in _MALE_EXCEPTIONS:
        return "male"
    if name.endswith(_FEMALE_ENDINGS):
        return "female"
    return None


_CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def transliterate(text: str) -> str:
    """Transliterate Cyrillic characters to Latin."""
    result_chars: list[str] = []
    for char in text:
        result_chars.append(_CYR_TO_LAT.get(char, char))
    return "".join(result_chars)
