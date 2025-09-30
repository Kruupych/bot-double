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


@dataclass(slots=True)
class ParticipantProfile:
    name: str
    samples: List[str]


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
        peers: Optional[List[ParticipantProfile]] = None,
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

        peer_section = ""
        if peers:
            peer_lines: List[str] = []
            for profile in peers:
                if not profile.samples:
                    continue
                joined = " / ".join(profile.samples)
                peer_lines.append(f"{profile.name}: {joined}")
            if peer_lines:
                peer_section = "Другие участники и их манера:\n" + "\n".join(peer_lines) + "\n\n"

        prompt = (
            f"Собери ответ в стиле пользователя {display_name}"
            f" (username: @{username}).\n\n"
            f"Примеры сообщений (без имен):\n{sample_block}\n\n"
            f"{context_section}"
            f"{peer_section}"
            f"Текст для ответа: {starter}\n\n"
            "Сформируй один ответ, как будто пишет сам пользователь."
            " Держи его тон, длину и структуру фраз."
            " Не добавляй перед ответом его имя или какие-либо префиксы."
            " Не цитируй дословно примеры или контекст — передавай смысл своими словами."
            " Не перечисляй несколько вариантов ответа."
            " Если в примерах нет эмодзи, не добавляй их."
            " Используй характерные словечки и интонации именно из примеров"
            " этого пользователя, а не общий шаблон."
            " Сделай ответ живым и конкретным по запросу."
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
