from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

WORD_RE = re.compile(r"[\w']+", re.UNICODE)

INFORMAL_PRONOUNS = {
    "ты",
    "тебя",
    "тебе",
    "тобой",
    "тобою",
    "твой",
    "твоя",
    "твои",
    "твоё",
    "твою",
    "твоих",
    "тебе",
}

FORMAL_PRONOUNS = {
    "вы",
    "ваш",
    "ваша",
    "ваши",
    "ваше",
    "вас",
    "вам",
    "вами",
}

TEASING_PATTERNS = (
    re.compile(r"\bаха+h?\b"),
    re.compile(r"\bхаха+\b"),
    re.compile(r"\bлол\b"),
    re.compile(r"\bприкалыва"),
    re.compile(r"\bподкалыва"),
    re.compile(r"\bнасмеш"),
    re.compile(r"[😂😅😜😉🥳🤪]"),
)


@dataclass(slots=True)
class InteractionSignals:
    informal: bool
    formal: bool
    teasing: bool

    def has_any(self) -> bool:
        return self.informal or self.formal or self.teasing


@dataclass(slots=True)
class RelationshipStats:
    total: int
    informal: int
    formal: int
    teasing: int
    samples: List[str]


def evaluate_interaction(text: str) -> InteractionSignals:
    lowered = text.lower()
    tokens = set(WORD_RE.findall(lowered))

    informal = any(token in INFORMAL_PRONOUNS for token in tokens)
    formal = any(token in FORMAL_PRONOUNS for token in tokens)
    teasing = any(pattern.search(lowered) for pattern in TEASING_PATTERNS)

    return InteractionSignals(informal=informal, formal=formal, teasing=teasing)


def build_relationship_hint(
    addressee_name: str, stats: RelationshipStats, min_samples: int = 3
) -> Optional[str]:
    if stats.total < min_samples:
        return None

    fragments: List[str] = []

    informal_ratio = stats.informal / stats.total if stats.total else 0.0
    formal_ratio = stats.formal / stats.total if stats.total else 0.0
    teasing_ratio = stats.teasing / stats.total if stats.total else 0.0

    if informal_ratio >= 0.4 and informal_ratio > formal_ratio:
        fragments.append("общается на 'ты'")
    elif formal_ratio >= 0.4 and formal_ratio > informal_ratio:
        fragments.append("держит дистанцию и использует 'вы'")

    if teasing_ratio >= 0.25:
        fragments.append("часто подшучивает")
    elif stats.teasing == 0 and informal_ratio < 0.2:
        fragments.append("общается сдержанно")

    if not fragments:
        return None

    summary = f"С {addressee_name} обычно {', '.join(fragments)}."

    sample_line = _pick_sample(stats.samples)
    if sample_line:
        summary += f" Пример: \"{sample_line}\""

    return summary


def _pick_sample(samples: Sequence[str]) -> Optional[str]:
    for text in samples[::-1]:
        trimmed = text.strip()
        if trimmed:
            return trimmed
    return None
