from __future__ import annotations

import io
import re
import unicodedata
from typing import List, Optional

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


# ============================================================================
# Long text handling for Telegram messages
# ============================================================================

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
SAFE_MESSAGE_LENGTH = 4000  # Leave some margin


def split_text_smart(text: str, max_len: int = SAFE_MESSAGE_LENGTH) -> List[str]:
    """
    Split long text into chunks, trying to break at paragraph or sentence boundaries.
    
    Priority for splitting:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentence endings (. ! ?)
    4. Any whitespace
    5. Hard cut at max_len (last resort)
    """
    if len(text) <= max_len:
        return [text]
    
    chunks: List[str] = []
    remaining = text
    
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        
        # Find the best split point within max_len
        chunk = remaining[:max_len]
        split_point = max_len
        
        # Try to find paragraph break (double newline)
        para_break = chunk.rfind("\n\n")
        if para_break > max_len // 3:  # At least 1/3 of max_len
            split_point = para_break + 2
        else:
            # Try single newline
            newline = chunk.rfind("\n")
            if newline > max_len // 3:
                split_point = newline + 1
            else:
                # Try sentence ending
                for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    pos = chunk.rfind(punct)
                    if pos > max_len // 3:
                        split_point = pos + len(punct)
                        break
                else:
                    # Try any whitespace
                    space = chunk.rfind(" ")
                    if space > max_len // 3:
                        split_point = space + 1
                    # Otherwise hard cut at max_len
        
        chunks.append(remaining[:split_point].rstrip())
        remaining = remaining[split_point:].lstrip()
    
    return [c for c in chunks if c]  # Filter out empty chunks


async def send_long_text(
    message: Message,
    text: str,
    *,
    max_message_len: int = SAFE_MESSAGE_LENGTH,
    document_threshold: int = 6000,
    document_filename: str = "text.txt",
    document_caption: Optional[str] = None,
) -> None:
    """
    Send potentially long text, handling Telegram's message limits.
    
    - If text <= max_message_len: sends as single message
    - If max_message_len < text <= document_threshold: splits into multiple messages
    - If text > document_threshold: sends as .txt document
    
    Args:
        message: Telegram message to reply to
        text: Text content to send
        max_message_len: Maximum length for a single message (default 4000)
        document_threshold: Length above which to send as document (default 6000)
        document_filename: Filename for document mode
        document_caption: Caption for document mode
    """
    text = text.strip()
    
    if not text:
        return
    
    # Simple case: fits in one message
    if len(text) <= max_message_len:
        await message.reply_text(text)
        return
    
    # Very long: send as document
    if len(text) > document_threshold:
        file_bytes = text.encode("utf-8")
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = document_filename
        await message.reply_document(
            document=file_obj,
            caption=document_caption or f"📄 Текст слишком длинный ({len(text)} символов)",
        )
        return
    
    # Medium: split into multiple messages
    chunks = split_text_smart(text, max_message_len)
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            await message.reply_text(chunk)
        else:
            # Send follow-up messages to the same chat
            if message.chat:
                await message.chat.send_message(chunk)
