from __future__ import annotations

import re
import unicodedata
from collections import Counter
from statistics import median
from typing import Iterable, List, Sequence

WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def build_style_summary(messages: Sequence[str]) -> str:
    cleaned = [text.strip() for text in messages if text and text.strip()]
    if not cleaned:
        return ""

    word_counts: List[int] = []
    letters_total = 0
    uppercase_letters = 0
    starts_with_upper = 0
    starts_with_lower = 0
    messages_with_exclaim = 0
    messages_with_question = 0
    messages_with_ellipsis = 0
    messages_with_formatting = 0
    messages_with_emoji = 0
    emoji_counter: Counter[str] = Counter()

    for text in cleaned:
        words = _tokenize(text)
        word_counts.append(len(words))

        first_alpha = _first_alpha(text)
        if first_alpha is not None:
            if first_alpha.isupper():
                starts_with_upper += 1
            elif first_alpha.islower():
                starts_with_lower += 1

        if "!" in text:
            messages_with_exclaim += 1
        if "?" in text:
            messages_with_question += 1
        if "..." in text:
            messages_with_ellipsis += 1
        if any(marker in text for marker in ("*", "_", "~", "`")):
            messages_with_formatting += 1

        emojis = _extract_emojis(text)
        if emojis:
            messages_with_emoji += 1
            emoji_counter.update(emojis)

        for char in text:
            if char.isalpha():
                letters_total += 1
                if char.isupper():
                    uppercase_letters += 1

    observations: List[str] = []

    avg_words = sum(word_counts) / len(word_counts)
    median_words = median(word_counts)
    observations.append(_describe_message_length(avg_words, median_words))

    if letters_total:
        caps_ratio = uppercase_letters / letters_total
        if caps_ratio >= 0.25:
            observations.append("часто использует заглавные буквы")
        elif caps_ratio <= 0.05:
            observations.append("предпочитает строчные буквы")

    total_with_alpha = starts_with_upper + starts_with_lower
    if total_with_alpha:
        start_upper_ratio = starts_with_upper / total_with_alpha
        if start_upper_ratio >= 0.8:
            observations.append("почти всегда начинает реплики с заглавной буквы")
        elif start_upper_ratio <= 0.2:
            observations.append("часто начинает сообщения с маленькой буквы")

    exclaim_ratio = messages_with_exclaim / len(cleaned)
    question_ratio = messages_with_question / len(cleaned)
    ellipsis_ratio = messages_with_ellipsis / len(cleaned)

    if exclaim_ratio >= 0.35:
        observations.append("эмоционально отвечает, любит восклицательные знаки")
    elif exclaim_ratio <= 0.05 and messages_with_exclaim == 0:
        observations.append("практически не использует восклицания")

    if question_ratio >= 0.35:
        observations.append("часто задаёт вопросы")

    if ellipsis_ratio >= 0.25:
        observations.append("любит многоточия")

    formatting_ratio = messages_with_formatting / len(cleaned)
    if formatting_ratio >= 0.15:
        observations.append("использует форматирование (жирный/курсив)")

    if messages_with_emoji:
        emoji_ratio = messages_with_emoji / len(cleaned)
        favourite = _format_favourite_emojis(emoji_counter)
        if favourite:
            if emoji_ratio >= 0.4:
                observations.append(f"часто добавляет эмодзи {favourite}")
            else:
                observations.append(f"иногда добавляет эмодзи {favourite}")
    else:
        observations.append("обычно пишет без эмодзи")

    observations = _deduplicate(observations)
    max_items = 6
    trimmed = observations[:max_items]
    bullets = "\n".join(f"- {item}" for item in trimmed)
    return bullets


def _tokenize(text: str) -> List[str]:
    return [token for token in WORD_RE.findall(text.lower()) if token]


def _first_alpha(text: str) -> str | None:
    for char in text:
        if char.isalpha():
            return char
    return None


def _extract_emojis(text: str) -> List[str]:
    emojis: List[str] = []
    for char in text:
        if _is_emoji(char):
            emojis.append(char)
    return emojis


def _is_emoji(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    if 0x1F300 <= codepoint <= 0x1FAFF:
        return True
    if 0x2600 <= codepoint <= 0x27BF:
        return True
    if 0x1F900 <= codepoint <= 0x1F9FF:
        return True
    category = unicodedata.category(char)
    return category in {"So", "Sk"}


def _describe_message_length(avg_words: float, med_words: float) -> str:
    avg = max(avg_words, 0.0)
    rounded = max(int(round(avg)), 1)
    if avg <= 8:
        return f"короткие сообщения (~{rounded} слов)"
    if avg <= 16:
        if abs(avg - med_words) <= 2:
            return f"средняя длина сообщения - около {rounded} слов"
        return f"сообщения средней длины (в среднем {rounded} слов)"
    return f"развёрнутые тексты (примерно {rounded} слов)"


def _format_favourite_emojis(counter: Counter[str]) -> str:
    if not counter:
        return ""
    most_common = counter.most_common(3)
    emojis = [emoji for emoji, _ in most_common]
    return " ".join(emojis)


def _deduplicate(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
