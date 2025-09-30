from __future__ import annotations

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
