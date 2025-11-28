from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI

_log = logging.getLogger(__name__)

from .prompts import (
    BATTLE_INSTRUCTIONS,
    BATTLE_SYSTEM,
    COMPATIBILITY_INSTRUCTIONS,
    COMPATIBILITY_SYSTEM,
    DIALOGUE_RULES_TEMPLATE,
    GENDER_FEMALE_INSTRUCTION,
    GENDER_MALE_INSTRUCTION,
    HOROSCOPE_INSTRUCTIONS,
    HOROSCOPE_SYSTEM,
    IMITATION_RULES,
    IMITATION_SYSTEM,
    NEWS_INSTRUCTIONS,
    NEWS_SYSTEM,
    REQUESTER_OTHER_INSTRUCTION,
    REQUESTER_SELF_INSTRUCTION,
    ROAST_INSTRUCTIONS,
    ROAST_SYSTEM,
    STORY_LONG_INSTRUCTIONS,
    STORY_SHORT_INSTRUCTIONS,
    STORY_SYSTEM,
    SUMMARY_INSTRUCTIONS,
    SUMMARY_SYSTEM,
    TINDER_INSTRUCTIONS,
    TINDER_SYSTEM,
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
    aliases: Optional[List[str]] = None


@dataclass(slots=True)
class DialogueParticipant:
    username: str
    name: str
    samples: List[StyleSample]
    style_summary: Optional[str]
    persona_card: Optional[str]
    relationship_hint: Optional[str]
    aliases: Optional[List[str]] = None


@dataclass(slots=True)
class StoryCharacter:
    """Character data for story generation."""
    name: str
    username: str
    samples: List[StyleSample]
    style_summary: Optional[str] = None
    aliases: Optional[List[str]] = None


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
        persona_aliases: Optional[List[str]] = None,
    ) -> str:
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for style generation")

        # Build aliases hint for persona
        aliases_hint = ""
        if persona_aliases:
            aliases_hint = f" (также известен как: {', '.join(persona_aliases)})"

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
            # Build requester aliases hint (only if not same person)
            requester_aliases_hint = ""
            if requester.aliases and not requester.is_same_person:
                requester_aliases_hint = f" (также известен как: {', '.join(requester.aliases)})"
            sample_hint = ""
            if requester.samples:
                joined = " / ".join(requester.samples)
                sample_hint = f" Он обычно пишет так: {joined}."
            requester_section = f"Вопрос задаёт {requester.name}{requester_aliases_hint}.{sample_hint}\n"
            if requester.is_same_person:
                _log.info("Adding SELF instruction for user %s", requester.name)
                requester_section += REQUESTER_SELF_INSTRUCTION
            else:
                requester_section += REQUESTER_OTHER_INSTRUCTION
            requester_section += "\n\n"

        rules_section = IMITATION_RULES + "\n\n"

        prompt = (
            f"Собери ответ в стиле пользователя {display_name}{aliases_hint}"
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

        def _aliases_hint(aliases: Optional[List[str]]) -> str:
            return f" (также известен как: {', '.join(aliases)})" if aliases else ""

        participant_sections = (
            f"{participant_a.name}{_aliases_hint(participant_a.aliases)} (@{participant_a.username}):\n{_participant_section(participant_a)}\n\n"
            f"{participant_b.name}{_aliases_hint(participant_b.aliases)} (@{participant_b.username}):\n{_participant_section(participant_b)}"
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
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Generate a playful roast based on user's communication style."""
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for roast generation")

        aliases_hint = ""
        if aliases:
            aliases_hint = f" (также известен как: {', '.join(aliases)})"

        summary_section = ""
        if style_summary:
            summary_section = f"\nХарактеристика стиля:\n{style_summary}\n"

        persona_section = ""
        if persona_card:
            persona_section = f"\nКарточка персоны:\n{persona_card}\n"

        prompt = (
            f"Сделай жёсткую прожарку пользователя {display_name}{aliases_hint} (@{username}).\n\n"
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
        aliases_a: Optional[List[str]],
        name_b: str,
        username_b: str,
        samples_b: Iterable[StyleSample],
        style_summary_b: Optional[str],
        aliases_b: Optional[List[str]],
    ) -> str:
        """Generate a fun compatibility analysis between two users."""
        sample_block_a = "\n".join(f"- {s.text}" for s in samples_a)
        sample_block_b = "\n".join(f"- {s.text}" for s in samples_b)

        if not sample_block_a or not sample_block_b:
            raise ValueError("Both users must have samples for compatibility check")

        aliases_hint_a = f" (также известен как: {', '.join(aliases_a)})" if aliases_a else ""
        aliases_hint_b = f" (также известен как: {', '.join(aliases_b)})" if aliases_b else ""

        summary_a = f"\nСтиль: {style_summary_a}" if style_summary_a else ""
        summary_b = f"\nСтиль: {style_summary_b}" if style_summary_b else ""

        instructions = COMPATIBILITY_INSTRUCTIONS.replace("{name_a}", name_a).replace("{name_b}", name_b)

        prompt = (
            f"Составь шуточный тест совместимости для {name_a} и {name_b}.\n\n"
            f"👤 {name_a}{aliases_hint_a} (@{username_a}):\n"
            f"Примеры сообщений:\n{sample_block_a}{summary_a}\n\n"
            f"👤 {name_b}{aliases_hint_b} (@{username_b}):\n"
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

    def generate_story(
        self,
        characters: List[StoryCharacter],
        topic: Optional[str] = None,
        *,
        long_version: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Generate a short story with chat participants as characters.
        
        Args:
            characters: List of StoryCharacter with their speech samples
            topic: Optional story theme/setting
            long_version: If True, generate longer story (4000-6000 chars)
            reasoning_effort: Override reasoning effort for this call
        """
        if len(characters) < 2:
            raise ValueError("Story needs at least 2 characters")
        
        # Build character descriptions
        char_sections: List[str] = []
        for char in characters:
            sample_block = "\n".join(f"- {s.text}" for s in char.samples)
            if not sample_block:
                continue
            aliases_hint = f" (также известен как: {', '.join(char.aliases)})" if char.aliases else ""
            section = f"👤 {char.name}{aliases_hint} (@{char.username}):\n"
            section += f"Примеры речи:\n{sample_block}"
            if char.style_summary:
                section += f"\nХарактеристика стиля: {char.style_summary}"
            char_sections.append(section)
        
        if len(char_sections) < 2:
            raise ValueError("Not enough characters with samples for story")
        
        character_names = ", ".join(c.name for c in characters)
        topic_text = topic.strip() if topic else "на усмотрение автора"
        
        instructions = STORY_LONG_INSTRUCTIONS if long_version else STORY_SHORT_INSTRUCTIONS
        
        prompt = (
            f"Напиши рассказ с персонажами: {character_names}.\n"
            f"Тема/сеттинг: {topic_text}.\n\n"
            "Персонажи и их стиль общения:\n\n"
            + "\n\n".join(char_sections)
            + f"\n\n{instructions}"
        )
        
        kwargs: dict[str, object] = {}
        # Use provided reasoning_effort, fall back to instance default
        effort = reasoning_effort or self._reasoning_effort
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        if self._text_verbosity:
            kwargs["text"] = {"verbosity": self._text_verbosity}
        
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": STORY_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()

    def generate_horoscope(
        self,
        username: str,
        display_name: str,
        samples: Iterable[StyleSample],
        style_summary: Optional[str] = None,
        persona_card: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Generate a personalized horoscope based on user's communication patterns."""
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for horoscope generation")

        aliases_hint = ""
        if aliases:
            aliases_hint = f" (также известен как: {', '.join(aliases)})"

        summary_section = ""
        if style_summary:
            summary_section = f"\nХарактеристика стиля:\n{style_summary}\n"

        persona_section = ""
        if persona_card:
            persona_section = f"\nКарточка персоны:\n{persona_card}\n"

        prompt = (
            f"Составь персональный гороскоп для {display_name}{aliases_hint} (@{username}).\n\n"
            f"Примеры его сообщений:\n{sample_block}\n"
            f"{summary_section}"
            f"{persona_section}\n"
            f"{HOROSCOPE_INSTRUCTIONS}"
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
                    "content": HOROSCOPE_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()

    def generate_battle(
        self,
        name_a: str,
        username_a: str,
        samples_a: Iterable[StyleSample],
        style_summary_a: Optional[str],
        aliases_a: Optional[List[str]],
        name_b: str,
        username_b: str,
        samples_b: Iterable[StyleSample],
        style_summary_b: Optional[str],
        aliases_b: Optional[List[str]],
    ) -> str:
        """Generate a rap battle between two users based on their communication styles."""
        sample_block_a = "\n".join(f"- {s.text}" for s in samples_a)
        sample_block_b = "\n".join(f"- {s.text}" for s in samples_b)

        if not sample_block_a or not sample_block_b:
            raise ValueError("Both users must have samples for battle")

        aliases_hint_a = f" (также известен как: {', '.join(aliases_a)})" if aliases_a else ""
        aliases_hint_b = f" (также известен как: {', '.join(aliases_b)})" if aliases_b else ""

        summary_a = f"\nСтиль: {style_summary_a}" if style_summary_a else ""
        summary_b = f"\nСтиль: {style_summary_b}" if style_summary_b else ""

        instructions = BATTLE_INSTRUCTIONS.replace("{name_a}", name_a).replace("{name_b}", name_b)

        prompt = (
            f"Организуй рэп-баттл между {name_a} и {name_b}.\n\n"
            f"🎤 {name_a}{aliases_hint_a} (@{username_a}):\n"
            f"Примеры сообщений:\n{sample_block_a}{summary_a}\n\n"
            f"🎤 {name_b}{aliases_hint_b} (@{username_b}):\n"
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
                    "content": BATTLE_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()

    def generate_news(
        self,
        chat_title: Optional[str],
        messages: List[dict],
    ) -> str:
        """
        Generate a tabloid-style news article about recent chat events.
        
        Args:
            chat_title: Name of the chat (if available)
            messages: List of recent messages with 'author' and 'text' keys
        """
        if not messages:
            raise ValueError("No messages supplied for news generation")

        chat_name = chat_title or "Секретный чат"
        
        # Format messages for the prompt
        message_lines: List[str] = []
        for msg in messages:
            author = msg.get("author", "Аноним")
            text = msg.get("text", "")
            if text:
                message_lines.append(f"{author}: {text}")
        
        if not message_lines:
            raise ValueError("No valid messages for news generation")
        
        messages_block = "\n".join(message_lines[-50:])  # Last 50 messages max
        
        prompt = (
            f"Напиши выпуск новостей для чата «{chat_name}».\n\n"
            f"Последние сообщения в чате:\n{messages_block}\n\n"
            f"{NEWS_INSTRUCTIONS}"
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
                    "content": NEWS_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()

    def generate_summary(
        self,
        chat_title: Optional[str],
        messages: List[dict],
    ) -> str:
        """
        Generate a factual summary of recent chat events.
        
        Args:
            chat_title: Name of the chat (if available)
            messages: List of recent messages with 'author' and 'text' keys
        """
        if not messages:
            raise ValueError("No messages supplied for summary generation")

        chat_name = chat_title or "чат"
        
        # Format messages for the prompt
        message_lines: List[str] = []
        for msg in messages:
            author = msg.get("author", "Аноним")
            text = msg.get("text", "")
            if text:
                message_lines.append(f"{author}: {text}")
        
        if not message_lines:
            raise ValueError("No valid messages for summary generation")
        
        messages_block = "\n".join(message_lines[-50:])  # Last 50 messages max
        
        prompt = (
            f"Составь резюме для чата «{chat_name}».\n\n"
            f"Последние сообщения:\n{messages_block}\n\n"
            f"{SUMMARY_INSTRUCTIONS}"
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
                    "content": SUMMARY_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()

    def generate_tinder(
        self,
        username: str,
        display_name: str,
        samples: Iterable[StyleSample],
        style_summary: Optional[str] = None,
        persona_card: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Generate a Tinder profile based on user's communication style."""
        sample_block = "\n".join(f"- {sample.text}" for sample in samples)
        if not sample_block:
            raise ValueError("No samples supplied for Tinder profile generation")

        aliases_hint = ""
        if aliases:
            aliases_hint = f" (также известен как: {', '.join(aliases)})"

        summary_section = ""
        if style_summary:
            summary_section = f"\nХарактеристика стиля:\n{style_summary}\n"

        persona_section = ""
        if persona_card:
            persona_section = f"\nКарточка персоны:\n{persona_card}\n"

        prompt = (
            f"Создай Tinder-профиль для {display_name}{aliases_hint} (@{username}).\n\n"
            f"Примеры его сообщений:\n{sample_block}\n"
            f"{summary_section}"
            f"{persona_section}\n"
            f"{TINDER_INSTRUCTIONS}"
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
                    "content": TINDER_SYSTEM,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,
        )
        return response.output_text.strip()
