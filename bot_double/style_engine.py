from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = (
    "Ты — бот-имитатор. По предоставленным фрагментам переписки ты должен отвечать в"
    " стиле указанного пользователя. Поддерживай тон, лексику, любимые выражения,"
    " эмодзи и типичный объём сообщения. Если собеседник часто шутит или ведёт"
    " себя неформально, отражай это. Отвечай на русском языке, если не указано иное."
    " Сохраняй естественный, непринуждённый стиль без скованности и роботизированных"
    " конструкций."
)


@dataclass(slots=True)
class StyleSample:
    text: str


@dataclass(slots=True)
class ContextMessage:
    speaker: str
    text: str


class StyleEngine:
    def __init__(self, api_key: str, model: str = "gpt-5-nano") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate_reply(
        self,
        username: str,
        display_name: str,
        samples: Iterable[StyleSample],
        starter: str,
        context: Optional[List[ContextMessage]] = None,
    ) -> str:
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for style generation")

        context_section = ""
        if context:
            context_lines = "\n".join(
                f"{message.speaker}: {message.text}" for message in context
            )
            context_section = f"Контекст диалога:\n{context_lines}\n\n"

        prompt = (
            f"Собери ответ в стиле пользователя {display_name}"
            f" (username: @{username}).\n\n"
            f"Примеры сообщений (без имен):\n{sample_block}\n\n"
            f"{context_section}"
            f"Текст для ответа: {starter}\n\n"
            "Сформируй один ответ, как будто пишет сам пользователь."
            " Держи его тон, длину и структуру фраз."
            " Не добавляй перед ответом его имя или какие-либо префиксы."
            " Не цитируй дословно примеры или контекст — передавай смысл своими словами."
            " Не перечисляй несколько вариантов ответа."
            " Если стиль предполагает эмодзи, вставь их так же естественно, как в примерах."
            " Ответ:"
        )

        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.output_text.strip()
