from __future__ import annotations

import difflib
import re
import sqlite3
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar

from .db import Database
from .utils import display_name, normalize_alias, transliterate

T = TypeVar("T")
RunDB = Callable[..., Awaitable[T]]

# Thresholds for descriptor matching
SCORE_EXACT_MATCH = 1.0
SCORE_SUBSTRING_MATCH = 0.92
SCORE_TOKEN_SUBSET = 0.88
SCORE_TOKEN_OVERLAP_BASE = 0.65
SCORE_TOKEN_OVERLAP_FACTOR = 0.25
SCORE_SIMILARITY_THRESHOLD = 0.75
SCORE_SIMILARITY_FACTOR = 0.85
SCORE_MINIMUM_THRESHOLD = 0.55
SCORE_CONFIDENCE_THRESHOLD = 0.78
SCORE_TOLERANCE = 0.05


class UserResolverService:
    """Service for resolving user descriptors to database user rows."""

    def __init__(
        self,
        *,
        db: Database,
        run_db: RunDB,
        get_bot_user_id: Callable[[], Optional[int]],
    ) -> None:
        self._db = db
        self._run_db = run_db
        self._get_bot_user_id = get_bot_user_id
        self._alias_cache: Dict[int, Dict[str, int]] = {}
        self._alias_display_cache: Dict[int, Dict[int, List[str]]] = {}

    def invalidate_cache(self, chat_id: Optional[int]) -> None:
        """Invalidate alias caches for a specific chat or all chats."""
        if chat_id is None:
            return
        self._alias_cache.pop(chat_id, None)
        self._alias_display_cache.pop(chat_id, None)

    def clear_all_caches(self) -> None:
        """Clear all alias caches."""
        self._alias_cache.clear()
        self._alias_display_cache.clear()

    async def get_alias_maps(
        self, chat_id: int
    ) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        """Get alias mappings for a chat (with caching)."""
        if chat_id in self._alias_cache and chat_id in self._alias_display_cache:
            return self._alias_cache[chat_id], self._alias_display_cache[chat_id]
        rows = await self._run_db(self._db.get_aliases_for_chat, chat_id)
        alias_map: Dict[str, int] = {}
        display_map: Dict[int, List[str]] = {}
        for row in rows:
            normalized = str(row["normalized_alias"])
            user_id = int(row["user_id"])
            alias_map[normalized] = user_id
            display_map.setdefault(user_id, []).append(str(row["alias"]))
        self._alias_cache[chat_id] = alias_map
        self._alias_display_cache[chat_id] = display_map
        return alias_map, display_map

    async def resolve_user_descriptor(
        self, chat_id: Optional[int], descriptor: str
    ) -> Tuple[Optional[sqlite3.Row], List[Tuple[sqlite3.Row, float]]]:
        """
        Resolve a user descriptor to a database row.
        
        Returns:
            A tuple of (resolved_row, suggestions).
            If a unique match is found, resolved_row is set and suggestions is empty.
            If multiple possible matches exist, resolved_row is None and suggestions contains candidates.
        """
        if not descriptor:
            return None, []
        descriptor = descriptor.strip()
        if descriptor.startswith("@"):
            username = descriptor.lstrip("@")
            row = await self._run_db(self._db.get_user_by_username, username)
            return row, []
        if chat_id is None:
            return None, []

        alias_map, alias_display = await self.get_alias_maps(chat_id)
        descriptor_variants = self._descriptor_variants(descriptor)

        for variant in list(descriptor_variants):
            if variant in alias_map:
                user_id = alias_map[variant]
                row = await self._run_db(self._db.get_user_by_id, user_id)
                if row:
                    return row, []

        participants = await self._run_db(self._db.get_chat_participants, chat_id)
        rows_by_id: Dict[int, sqlite3.Row] = {
            int(row["id"]): row for row in participants
        }
        # include aliased users who may not have messages yet
        for user_id in alias_display.keys():
            if user_id not in rows_by_id:
                row = await self._run_db(self._db.get_user_by_id, user_id)
                if row:
                    rows_by_id[user_id] = row

        bot_user_id = self._get_bot_user_id()
        best_candidates: List[Tuple[sqlite3.Row, float]] = []
        best_score = 0.0
        for user_id, row in rows_by_id.items():
            if bot_user_id is not None and user_id == bot_user_id:
                continue
            aliases = alias_display.get(user_id)
            candidate_keys = self._candidate_keys_for_user(row, aliases)
            score = self._score_descriptor(descriptor_variants, candidate_keys)
            if score < SCORE_MINIMUM_THRESHOLD:
                continue
            if score > best_score + SCORE_TOLERANCE:
                best_score = score
                best_candidates = [(row, score)]
            elif abs(score - best_score) <= SCORE_TOLERANCE:
                best_candidates.append((row, score))

        if best_score >= SCORE_CONFIDENCE_THRESHOLD and len(best_candidates) == 1:
            return best_candidates[0][0], []

        if best_candidates:
            best_candidates.sort(key=lambda item: item[1], reverse=True)
            return None, best_candidates[:3]

        return None, []

    def _generate_variants(self, values: Iterable[str]) -> Set[str]:
        """Generate normalized and transliterated variants for matching."""
        variants: Set[str] = set()
        for value in values:
            normalized = normalize_alias(value)
            if not normalized:
                continue
            variants.add(normalized)
            translit = transliterate(normalized)
            if translit and translit != normalized:
                variants.add(translit)
        return variants

    def _candidate_keys_for_user(
        self, row: sqlite3.Row, aliases: Optional[List[str]]
    ) -> Set[str]:
        """Generate candidate keys for a user row."""
        values: List[str] = []
        username = row["username"]
        first_name = row["first_name"]
        last_name = row["last_name"]
        if username:
            values.append(username)
            values.append(f"@{username}")
        if first_name:
            values.append(first_name)
        if last_name:
            values.append(last_name)
        if first_name and last_name:
            values.append(f"{first_name} {last_name}")
        display = display_name(username, first_name, last_name)
        if display:
            values.append(display)
        if aliases:
            values.extend(aliases)
        return self._generate_variants(values)

    def _descriptor_variants(self, descriptor: str) -> Set[str]:
        """Generate matching variants for a descriptor string."""
        cleaned = descriptor.strip()
        cleaned = re.sub(r"[\"'""«»]+", "", cleaned)
        candidates: List[str] = []
        parts = re.split(r"[:;,\n\-–—]", cleaned, maxsplit=1)
        base = parts[0].strip()
        if base:
            candidates.append(base)
        tokens = base.split()
        if len(tokens) > 1:
            for length in range(len(tokens), 0, -1):
                segment = " ".join(tokens[:length]).strip()
                if segment and segment not in candidates:
                    candidates.append(segment)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
        variants: Set[str] = set()
        for candidate in candidates:
            normalized = normalize_alias(candidate)
            if not normalized:
                continue
            variants.add(normalized)
            translit = transliterate(normalized)
            if translit and translit != normalized:
                variants.add(translit)
        return variants

    def _score_descriptor(
        self, descriptor_variants: Iterable[str], candidate_keys: Iterable[str]
    ) -> float:
        """Score how well descriptor variants match candidate keys."""
        best = 0.0
        candidate_list = list(candidate_keys)
        if not candidate_list:
            return 0.0
        for descriptor in descriptor_variants:
            tokens1 = set(descriptor.split())
            for candidate in candidate_list:
                if descriptor == candidate:
                    return SCORE_EXACT_MATCH
                if descriptor and candidate and (
                    descriptor in candidate or candidate in descriptor
                ):
                    best = max(best, SCORE_SUBSTRING_MATCH)
                    continue
                tokens2 = set(candidate.split())
                if tokens1 and tokens1 <= tokens2:
                    best = max(best, SCORE_TOKEN_SUBSET)
                    continue
                if tokens1 and tokens2:
                    overlap = tokens1 & tokens2
                    if overlap:
                        ratio = len(overlap) / max(len(tokens1), len(tokens2))
                        best = max(best, SCORE_TOKEN_OVERLAP_BASE + SCORE_TOKEN_OVERLAP_FACTOR * ratio)
                        continue
                similarity = difflib.SequenceMatcher(None, descriptor, candidate).ratio()
                if similarity >= SCORE_SIMILARITY_THRESHOLD:
                    best = max(best, similarity * SCORE_SIMILARITY_FACTOR)
        return best

