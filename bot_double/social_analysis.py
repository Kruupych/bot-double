from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from openai import OpenAI


ANALYSIS_SYSTEM_PROMPT = (
    "Ты — эксперт по социальной психологии и анализу коммуникаций."
    " Твоя задача — внимательно читать диалог и делать взвешенные выводы"
    " о тоне, степени тепла и динамике отношений между собеседниками."
    " Формируй ответы только на основании приведённого текста, без домыслов."
)


@dataclass(slots=True)
class InteractionExcerpt:
    focus_text: str
    focus_timestamp: int
    context: Sequence[str]


class SocialAnalyzer:
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

    def analyze_relationship(
        self,
        speaker_name: str,
        target_name: str,
        excerpts: Iterable[InteractionExcerpt],
    ) -> Optional[dict[str, object]]:
        blocks: List[str] = []
        for index, excerpt in enumerate(excerpts, start=1):
            context_section = "\n".join(excerpt.context)
            blocks.append(
                "\n".join(
                    [
                        f"### Эпизод {index}",
                        f"Время: {excerpt.focus_timestamp}",
                        "Контекст:",
                        context_section or "(контекст отсутствует)",
                        "Реплика:",
                        f"{speaker_name}: {excerpt.focus_text}",
                    ]
                )
            )

        if not blocks:
            return None

        dialogue_block = "\n\n".join(blocks)

        instructions = (
            "Проанализируй манеру общения одной стороны по отношению к другой."
            f" Говорящий: {speaker_name}. Адресат: {target_name}."
            " Используй приведённые эпизоды как материал."
            " Верни результат строго в формате JSON со следующими полями:"
            " {\"summary\": string, \"tone\": string, \"formality\": string,"
            " \"teasing_level\": string, \"respect_level\": string,"
            " \"emotional_notes\": string, \"example_quotes\": array}."
            " \"example_quotes\" — список до трёх характерных цитат говорящего."
            " Если какие-то поля оценить невозможно, заполни их строкой 'unknown'."
        )

        prompt = (
            f"{instructions}\n\n"
            f"Диалог:\n{dialogue_block}\n\n"
            "Ответ должен содержать только один JSON-объект без пояснений."
        )

        kwargs: dict[str, object] = {}
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}

        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
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
