from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, List, Optional, Tuple, TypeVar

from .analysis_worker import AnalysisWorker
from .config import Settings
from .db import Database
from .social_analysis import InteractionExcerpt, SocialAnalyzer
from .utils import display_name

T = TypeVar("T")
RunDB = Callable[..., Awaitable[T]]

# Key type: (chat_id, speaker_id, target_id)
RelationshipKey = Tuple[int, int, int]


class RelationshipAnalysisWorker(AnalysisWorker[RelationshipKey]):
    """Worker for analyzing relationships between users."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        run_db: RunDB,
        social_analyzer: SocialAnalyzer,
    ) -> None:
        super().__init__(name="Relationship")
        self._settings = settings
        self._db = db
        self._run_db = run_db
        self._social = social_analyzer

    async def _should_queue(self, key: RelationshipKey) -> bool:
        chat_id, speaker_id, target_id = key
        stats = await self._run_db(
            self._db.get_pair_stats,
            chat_id,
            speaker_id,
            target_id,
        )
        if not stats:
            return False
        pending = int(stats.get("pending_messages", 0))
        if pending < self._settings.relationship_analysis_min_pending:
            return False
        last_analyzed = stats.get("last_analyzed_at")
        threshold_seconds = self._settings.relationship_analysis_min_hours * 3600
        if (
            threshold_seconds > 0
            and last_analyzed is not None
            and int(time.time()) - int(last_analyzed) < threshold_seconds
        ):
            return False
        return True

    async def _perform_analysis(self, key: RelationshipKey) -> None:
        chat_id, speaker_id, target_id = key
        stats = await self._run_db(
            self._db.get_pair_stats,
            chat_id,
            speaker_id,
            target_id,
        )
        if not stats:
            return
        pending = int(stats.get("pending_messages", 0))
        if pending < self._settings.relationship_analysis_min_pending:
            return

        speaker_row = await self._run_db(self._db.get_user_by_id, speaker_id)
        target_row = await self._run_db(self._db.get_user_by_id, target_id)
        if speaker_row is None or target_row is None:
            return

        speaker_name = display_name(
            speaker_row["username"],
            speaker_row["first_name"],
            speaker_row["last_name"],
        )
        target_name = display_name(
            target_row["username"],
            target_row["first_name"],
            target_row["last_name"],
        )

        max_samples = 20
        rows = await self._run_db(
            self._db.get_pair_messages,
            chat_id,
            speaker_id,
            target_id,
            max_samples,
        )
        if not rows:
            return

        min_required = max(
            self._settings.relationship_analysis_min_pending, 5
        )
        if len(rows) < min_required:
            return

        excerpts: List[InteractionExcerpt] = []
        for record in reversed(rows):
            focus_text = (record["text"] or "").strip()
            if not focus_text:
                continue
            focus_timestamp = int(record["timestamp"])
            context_rows = await self._run_db(
                self._db.get_chat_messages_before,
                chat_id,
                focus_timestamp,
                3,
            )
            context_lines: List[str] = []
            for ctx in context_rows[-3:]:
                ctx_text = (ctx["text"] or "").strip()
                if not ctx_text:
                    continue
                ctx_name = display_name(
                    ctx["username"], ctx["first_name"], ctx["last_name"]
                )
                context_lines.append(f"{ctx_name}: {ctx_text}")
            excerpts.append(
                InteractionExcerpt(
                    focus_text=focus_text,
                    focus_timestamp=focus_timestamp,
                    context=tuple(context_lines),
                )
            )

        if len(excerpts) < min_required:
            return

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._social.analyze_relationship(
                speaker_name, target_name, tuple(excerpts)
            ),
        )
        if not result:
            return

        summary = str(result.get("summary", "")).strip()
        if not summary:
            tone = str(result.get("tone", "unknown")).strip()
            notes = str(result.get("emotional_notes", "unknown")).strip()
            summary = f"Тон общения: {tone}. Эмоциональные заметки: {notes}."
        details_json = json.dumps(result, ensure_ascii=False)
        analyzed_at = int(time.time())
        await self._run_db(
            self._db.save_pair_analysis,
            chat_id,
            speaker_id,
            target_id,
            summary,
            details_json,
            analyzed_at,
        )

