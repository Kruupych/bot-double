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
    # Strip markup and count words
    tokens = message.text.replace("\n", " ").split()
    if len(tokens) < min_tokens:
        return False
    lowered = message.text.lower()
    if "http://" in lowered or "https://" in lowered:
        return False
    return True
