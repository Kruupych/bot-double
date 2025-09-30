from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = (
    "Ты — бот-имитатор. По предоставленным фрагментам переписки ты должен отвечать"
    " в стиле указанного пользователя. Поддерживай тон, лексику, частоту эмодзи"
    " и манеру общения. Отвечай на русском языке, если не указано иное."
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
            " Не добавляй перед ответом его имя или какие-либо префиксы."
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
