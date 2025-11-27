from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

from .prompts import PERSONA_INSTRUCTIONS, PERSONA_SYSTEM


@dataclass(slots=True)
class PersonaSample:
    text: str
    timestamp: int


class PersonaAnalyzer:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._reasoning_effort = reasoning_effort

    def build_persona_card(
        self,
        display_name: str,
        samples: Iterable[PersonaSample],
    ) -> Optional[dict[str, object]]:
        collected: List[Tuple[int, str]] = []
        for sample in samples:
            text = (sample.text or "").strip()
            if not text:
                continue
            collected.append((sample.timestamp, text))

        if not collected:
            return None

        collected.sort()
        message_block = "\n".join(
            f"[{idx + 1}] {text}"
            for idx, (_, text) in enumerate(collected)
        )

        instructions = PERSONA_INSTRUCTIONS

        prompt = (
            f"{instructions}\n\n"
            f"Пользователь: {display_name}.\n"
            f"Сообщения:\n{message_block}\n\n"
            "Ответ должен содержать только один JSON-объект без пояснений."
        )

        kwargs: dict[str, object] = {}
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}

        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": PERSONA_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )

        text = response.output_text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data
