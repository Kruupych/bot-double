from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Callable, List, Optional, Tuple, TypeVar

from .analysis_worker import AnalysisWorker
from .config import Settings
from .db import Database
from .persona_analysis import PersonaAnalyzer, PersonaSample
from .utils import display_name

T = TypeVar("T")
RunDB = Callable[..., Awaitable[T]]

# Key type: (chat_id, user_id)
PersonaKey = Tuple[int, int]


class PersonaAnalysisWorker(AnalysisWorker[PersonaKey]):
    """Worker for analyzing user personas."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        run_db: RunDB,
        persona_analyzer: PersonaAnalyzer,
    ) -> None:
        super().__init__(name="Persona")
        self._settings = settings
        self._db = db
        self._run_db = run_db
        self._analyzer = persona_analyzer

    async def _should_queue(self, key: PersonaKey) -> bool:
        chat_id, user_id = key
        profile = await self._run_db(
            self._db.get_persona_profile,
            chat_id,
            user_id,
        )
        if not profile:
            return False
        pending = int(profile.get("pending_messages", 0))
        if pending < self._settings.persona_analysis_min_messages:
            return False
        last_analyzed = profile.get("last_analyzed_at")
        threshold_seconds = self._settings.persona_analysis_min_hours * 3600
        if (
            threshold_seconds > 0
            and last_analyzed is not None
            and int(time.time()) - int(last_analyzed) < threshold_seconds
        ):
            return False
        return True

    async def _perform_analysis(self, key: PersonaKey) -> None:
        chat_id, user_id = key
        profile = await self._run_db(
            self._db.get_persona_profile,
            chat_id,
            user_id,
        )
        if not profile:
            return
        pending = int(profile.get("pending_messages", 0))
        if pending < self._settings.persona_analysis_min_messages:
            return

        user_row = await self._run_db(self._db.get_user_by_id, user_id)
        if user_row is None:
            return

        messages = await self._run_db(
            self._db.get_recent_messages_with_timestamp,
            chat_id,
            user_id,
            self._settings.persona_analysis_max_messages,
        )
        if not messages:
            return

        samples: List[PersonaSample] = []
        for row in messages:
            text = (row["text"] or "").strip()
            if not text:
                continue
            samples.append(
                PersonaSample(text=text, timestamp=int(row["timestamp"]))
            )

        if len(samples) < self._settings.persona_analysis_min_messages:
            return

        user_display = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._analyzer.build_persona_card(
                user_display, tuple(samples)
            ),
        )
        if not result:
            return

        result.pop("writing_tips", None)

        summary = str(result.get("overall_summary", "")).strip()
        if not summary:
            interests = result.get("interests") or []
            if isinstance(interests, list) and interests:
                joined = ", ".join(str(item) for item in interests if item)
                if joined:
                    summary = f"Интересы: {joined}."
        if not summary:
            traits = result.get("speech_traits") or []
            if isinstance(traits, list) and traits:
                joined = ", ".join(str(item) for item in traits if item)
                if joined:
                    summary = f"Речевые привычки: {joined}."
        if not summary:
            summary = "Карточка персоны обновлена."

        details_json = json.dumps(result, ensure_ascii=False)
        analyzed_at = int(time.time())
        await self._run_db(
            self._db.save_persona_profile,
            chat_id,
            user_id,
            summary,
            details_json,
            analyzed_at,
        )

