from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI


PERSONA_SYSTEM_PROMPT = (
    "Ты — эксперт по психологии общения и стилю речи."
    " На основе выборки сообщений пользователя опиши его портрет."
    " Работай аккуратно и избегай неподтверждённых предположений."
)


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

        instructions = (
            "На основе реплик пользователя составь его краткую персональную карточку."
            " Отрази интересы, чувство юмора, эмоциональность, общий тон и характерные речевые привычки."
            " Делай выводы только из текстов, избегай стереотипов."
            " Верни структуру в формате JSON с полями:"
            " {\"overall_summary\": string, \"interests\": array,"
            " \"humor_style\": string, \"emotionality\": string, \"tonality\": string,"
            " \"speech_traits\": array}."
            " Массивы заполняй строками. Если данных не хватает, используй значение 'unknown'."
        )

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
                {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
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
