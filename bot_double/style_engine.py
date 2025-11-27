from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI

from .prompts import (
    COMPATIBILITY_INSTRUCTIONS,
    COMPATIBILITY_SYSTEM,
    DIALOGUE_RULES_TEMPLATE,
    GENDER_FEMALE_INSTRUCTION,
    GENDER_MALE_INSTRUCTION,
    IMITATION_RULES,
    IMITATION_SYSTEM,
    REQUESTER_OTHER_INSTRUCTION,
    REQUESTER_SELF_INSTRUCTION,
    ROAST_INSTRUCTIONS,
    ROAST_SYSTEM,
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
    persona_card: Optional[str]
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
        persona_card: Optional[str] = None,
        relationship_hint: Optional[str] = None,
    ) -> str:
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for style generation")

        summary_section = ""
        if style_summary:
            summary_section = f"Характеристика стиля:\n{style_summary}\n\n"

        persona_section = ""
        if persona_card:
            persona_section = f"Карточка персоны:\n{persona_card}\n\n"

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
                requester_section += REQUESTER_SELF_INSTRUCTION
            else:
                requester_section += REQUESTER_OTHER_INSTRUCTION
            requester_section += "\n\n"

        rules_section = IMITATION_RULES + "\n\n"

        prompt = (
            f"Собери ответ в стиле пользователя {display_name}"
            f" (username: @{username}).\n\n"
            f"Примеры сообщений (без имен):\n{sample_block}\n\n"
            f"{summary_section}"
            f"{persona_section}"
            f"{relationship_section}"
            f"{context_section}"
            f"{peer_section}"
            f"{requester_section}"
            f"{rules_section}"
            f"{self._gender_instruction(persona_gender)}"
            f"{starter.strip()}\n\n"
            "Ответ:"
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
                    "content": IMITATION_SYSTEM,
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
            return GENDER_FEMALE_INSTRUCTION + "\n\n"
        if gender == "male":
            return GENDER_MALE_INSTRUCTION + "\n\n"
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
            persona_card = participant.persona_card or ""
            relationship = participant.relationship_hint or ""
            parts: List[str] = [f"Примеры речи:\n{sample_block}"]
            if summary:
                parts.append(f"Стиль: {summary}")
            if persona_card:
                parts.append(f"Персона: {persona_card}")
            if relationship:
                parts.append(f"Особенности общения: {relationship}")
            return "\n".join(parts)

        participant_sections = (
            f"{participant_a.name} (@{participant_a.username}):\n{_participant_section(participant_a)}\n\n"
            f"{participant_b.name} (@{participant_b.username}):\n{_participant_section(participant_b)}"
        )

        dialogue_rules = DIALOGUE_RULES_TEMPLATE.format(
            participant_a=participant_a.name,
            participant_b=participant_b.name,
        )
        prompt = (
            "Составь короткий, связный диалог между двумя участниками чата."
            f" Тема беседы: {topic_text}.\n\n"
            "Участники и их стиль:\n"
            f"{participant_sections}\n\n"
            f"{dialogue_rules}"
        )

        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": IMITATION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.output_text.strip()

    def generate_roast(
        self,
        username: str,
        display_name: str,
        samples: Iterable[StyleSample],
        style_summary: Optional[str] = None,
        persona_card: Optional[str] = None,
    ) -> str:
        """Generate a playful roast based on user's communication style."""
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for roast generation")

        summary_section = ""
        if style_summary:
            summary_section = f"\nХарактеристика стиля:\n{style_summary}\n"

        persona_section = ""
        if persona_card:
            persona_section = f"\nКарточка персоны:\n{persona_card}\n"

        prompt = (
            f"Сделай добродушную «поджарку» пользователя {display_name} (@{username}).\n\n"
            f"Примеры его сообщений:\n{sample_block}\n"
            f"{summary_section}"
            f"{persona_section}\n"
            f"{ROAST_INSTRUCTIONS}"
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
                    "content": ROAST_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()

    def generate_compatibility(
        self,
        name_a: str,
        username_a: str,
        samples_a: Iterable[StyleSample],
        style_summary_a: Optional[str],
        name_b: str,
        username_b: str,
        samples_b: Iterable[StyleSample],
        style_summary_b: Optional[str],
    ) -> str:
        """Generate a fun compatibility analysis between two users."""
        sample_block_a = "\n".join(f"- {s.text}" for s in samples_a)
        sample_block_b = "\n".join(f"- {s.text}" for s in samples_b)

        if not sample_block_a or not sample_block_b:
            raise ValueError("Both users must have samples for compatibility check")

        summary_a = f"\nСтиль: {style_summary_a}" if style_summary_a else ""
        summary_b = f"\nСтиль: {style_summary_b}" if style_summary_b else ""

        instructions = COMPATIBILITY_INSTRUCTIONS.replace("{name_a}", name_a).replace("{name_b}", name_b)

        prompt = (
            f"Составь шуточный тест совместимости для {name_a} и {name_b}.\n\n"
            f"👤 {name_a} (@{username_a}):\n"
            f"Примеры сообщений:\n{sample_block_a}{summary_a}\n\n"
            f"👤 {name_b} (@{username_b}):\n"
            f"Примеры сообщений:\n{sample_block_b}{summary_b}\n\n"
            f"{instructions}"
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
                    "content": COMPATIBILITY_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()
