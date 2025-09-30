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


@dataclass(slots=True)
class RequesterProfile:
    name: str
    samples: List[str]
    is_same_person: bool


@dataclass(slots=True)
class DialogueParticipant:
    username: str
    name: str
    samples: List[StyleSample]
    style_summary: Optional[str]
    relationship_hint: Optional[str]


class StyleEngine:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-nano",
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
        persona_gender_hint: Optional[str] = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._text_verbosity = text_verbosity
        self._persona_gender_hint = persona_gender_hint

    def generate_reply(
        self,
        username: str,
        display_name: str,
        samples: Iterable[StyleSample],
        starter: str,
        context: Optional[List[ContextMessage]] = None,
        peers: Optional[List[ParticipantProfile]] = None,
        requester: Optional[RequesterProfile] = None,
        persona_gender: Optional[str] = None,
        style_summary: Optional[str] = None,
        relationship_hint: Optional[str] = None,
    ) -> str:
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for style generation")

        summary_section = ""
        if style_summary:
            summary_section = f"Характеристика стиля:\n{style_summary}\n\n"

        relationship_section = ""
        if relationship_hint:
            relationship_section = (
                f"Особенности отношений с адресатом:\n{relationship_hint}\n\n"
            )

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

        requester_section = ""
        if requester:
            sample_hint = ""
            if requester.samples:
                joined = " / ".join(requester.samples)
                sample_hint = f" Он обычно пишет так: {joined}."
            requester_section = f"Вопрос задаёт {requester.name}.{sample_hint}\n"
            if requester.is_same_person:
                requester_section += (
                    "Он спрашивает сам у себя. Ответ сформулируй как внутренний"
                    " монолог или честное признание, без обращения к себе как"
                    " к другому человеку."
                )
            else:
                requester_section += (
                    "Можешь обращаться к нему по имени или на 'ты', если это"
                    " соответствует стилю примеров."
                )
            requester_section += "\n\n"

        prompt = (
            f"Собери ответ в стиле пользователя {display_name}"
            f" (username: @{username}).\n\n"
            f"Примеры сообщений (без имен):\n{sample_block}\n\n"
            f"{summary_section}"
            f"{relationship_section}"
            f"{context_section}"
            f"{peer_section}"
            f"{requester_section}"
            f"{self._gender_instruction(persona_gender)}"
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
            " Строй реплику как цельный связный текст: добавляй плавные переходы,"
            " логично развивай мысль и держи один тон."
            " Избегай случайного перечисления отдельных фраз без связи."
            " Не повторяй вопрос собеседника дословно и не возвращай его в виде ответа."
            " Если вопрос короткий, обвинительный или провокационный, расширь ответ"
            " своей реакцией: добавь детали, эмоции или мысли, характерные для"
            " пользователя, вместо простого подтверждения."
            " Ответ:"
        )

        kwargs: dict[str, object] = {}
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        if self._text_verbosity:
            kwargs["text"] = {"verbosity": self._text_verbosity}

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
            **kwargs,
        )
        return response.output_text.strip()

    def _gender_instruction(self, persona_gender: Optional[str]) -> str:
        gender = persona_gender or self._persona_gender_hint
        if gender == "female":
            return (
                "Пользователь говорит от женского лица. Используй женские формы"
                " глаголов и местоимений, если это уместно." "\n\n"
            )
        if gender == "male":
            return (
                "Пользователь говорит от мужского лица. Используй мужские формы"
                " глаголов и местоимений, если это уместно." "\n\n"
            )
        return ""

    def generate_dialogue(
        self,
        participant_a: DialogueParticipant,
        participant_b: DialogueParticipant,
        topic: str,
    ) -> str:
        if not participant_a.samples or not participant_b.samples:
            raise ValueError("Both participants must have style samples for dialogue")

        topic_text = topic.strip() or "свободную тему"

        def _participant_section(participant: DialogueParticipant) -> str:
            sample_block = "\n".join(f"- {sample.text}" for sample in participant.samples)
            summary = participant.style_summary or ""
            relationship = participant.relationship_hint or ""
            parts: List[str] = [f"Примеры речи:\n{sample_block}"]
            if summary:
                parts.append(f"Стиль: {summary}")
            if relationship:
                parts.append(f"Отношения с собеседником: {relationship}")
            return "\n".join(parts)

        participant_sections = (
            f"{participant_a.name} (@{participant_a.username}):\n{_participant_section(participant_a)}\n\n"
            f"{participant_b.name} (@{participant_b.username}):\n{_participant_section(participant_b)}"
        )

        prompt = (
            "Составь короткий диалог между двумя участниками чата."
            f" Тема беседы: {topic_text}.\n\n"
            "Участники и их стиль:\n"
            f"{participant_sections}\n\n"
            "Сгенерируй 4-6 реплик, чередуя участников."
            " Каждый ответ пишется в одну строку в формате 'Имя: текст'."
            " Учитывай стиль каждого и указанные отношения."
            " Реплики должны быть естественными и развивать тему."
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
