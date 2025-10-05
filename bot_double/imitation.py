from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from telegram import Message, User

from .utils import display_name


@dataclass
class ChainMessage:
    speaker: str
    text: str
    is_persona: bool


@dataclass
class ImitationChain:
    chat_id: int
    persona_id: int
    persona_username: Optional[str]
    persona_first_name: Optional[str]
    persona_last_name: Optional[str]
    persona_name: str
    messages: List[ChainMessage]


class ImitationToolkit:
    def __init__(
        self,
        *,
        bot_name: str,
        bot_username: Optional[str],
        chain_cache_limit: int,
        answered_cache_limit: int,
    ) -> None:
        self._bot_name = bot_name
        self._bot_username = bot_username
        self._chain_cache_limit = chain_cache_limit
        self._answered_cache_limit = answered_cache_limit
        self._chain_by_message: Dict[Tuple[int, int], ImitationChain] = {}
        self._last_chain_by_user: Dict[Tuple[int, int], ImitationChain] = {}
        self._answered_messages: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------ identity ---
    def update_bot_identity(
        self, *, bot_name: Optional[str], bot_username: Optional[str]
    ) -> None:
        if bot_name:
            self._bot_name = bot_name
        if bot_username is not None:
            self._bot_username = bot_username

    # ---------------------------------------------------------------------- caches ---
    def _clone_chain(self, chain: ImitationChain) -> ImitationChain:
        return ImitationChain(
            chat_id=chain.chat_id,
            persona_id=chain.persona_id,
            persona_username=chain.persona_username,
            persona_first_name=chain.persona_first_name,
            persona_last_name=chain.persona_last_name,
            persona_name=chain.persona_name,
            messages=[
                ChainMessage(
                    speaker=entry.speaker,
                    text=entry.text,
                    is_persona=entry.is_persona,
                )
                for entry in chain.messages
            ],
        )

    def register_chain_reference(
        self, chat_id: int, message_id: int, chain: ImitationChain
    ) -> None:
        key = (chat_id, message_id)
        self._chain_by_message[key] = self._clone_chain(chain)
        if len(self._chain_by_message) > self._chain_cache_limit:
            oldest_key = next(iter(self._chain_by_message))
            if oldest_key != key:
                self._chain_by_message.pop(oldest_key, None)

    def remember_chain_for_user(
        self, chat_id: int, user_id: int, chain: ImitationChain
    ) -> None:
        key = (chat_id, user_id)
        self._last_chain_by_user[key] = self._clone_chain(chain)
        if len(self._last_chain_by_user) > self._chain_cache_limit:
            oldest_key = next(iter(self._last_chain_by_user))
            if oldest_key != key:
                self._last_chain_by_user.pop(oldest_key, None)

    def get_chain_for_message(
        self, chat_id: int, message_id: int
    ) -> Optional[ImitationChain]:
        chain = self._chain_by_message.get((chat_id, message_id))
        return self._clone_chain(chain) if chain else None

    def reserve_answer_slot(self, message: Message) -> bool:
        chat_id = message.chat_id
        message_id = getattr(message, "message_id", None)
        if chat_id is None or message_id is None:
            return True
        key = (chat_id, int(message_id))
        if key in self._answered_messages:
            return False
        self._answered_messages[key] = int(time.time())
        if len(self._answered_messages) > self._answered_cache_limit:
            oldest_key = next(iter(self._answered_messages))
            if oldest_key != key:
                self._answered_messages.pop(oldest_key, None)
        return True

    # ----------------------------------------------------------- text processing ---
    def strip_call_signs(self, text: str) -> str:
        punctuation = r":,;/\s\-–—"
        patterns = [
            rf"^бот[{punctuation}]*двойник[{punctuation}]*",
            rf"^двойник[{punctuation}]*бот[{punctuation}]*",
            rf"^бот[{punctuation}]*",
            rf"^двойник[{punctuation}]*",
        ]
        if self._bot_username:
            patterns.append(rf"@{re.escape(self._bot_username)}")
        if self._bot_name:
            patterns.append(re.escape(self._bot_name))
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def text_after_delimiter(self, text: str) -> Optional[str]:
        for delimiter in (":", "—", "–"):
            if delimiter in text:
                candidate = text.split(delimiter, 1)[1].strip()
                if candidate:
                    return candidate
        if "\n" in text:
            tail = text.split("\n", 1)[1].strip()
            if tail:
                return tail
        return None

    @staticmethod
    def text_in_quotes(text: str) -> Optional[str]:
        match = re.search(r'["«](.+?)["»]', text)
        if match:
            return match.group(1).strip()
        return None

    def extract_reply_text(self, message: Message) -> Optional[str]:
        reply = message.reply_to_message
        if reply is None:
            return None
        if reply.text:
            return reply.text.strip()
        if reply.caption:
            return reply.caption.strip()
        return None

    def extract_payload(
        self,
        message: Message,
        text: str,
        *,
        keywords: Optional[List[str]] = None,
    ) -> Optional[str]:
        inline = self.text_after_delimiter(text)
        if inline:
            return inline
        quoted = self.text_in_quotes(text)
        if quoted:
            return quoted
        cleaned = text
        if keywords:
            for keyword in keywords:
                pattern = re.compile(keyword, re.IGNORECASE)
                match = pattern.search(cleaned)
                if match:
                    cleaned = cleaned[match.end() :]
        cleaned = self.strip_call_signs(cleaned).strip(" ,\n-—")
        if cleaned:
            return cleaned
        reply_text = self.extract_reply_text(message)
        if reply_text:
            return reply_text
        return None

    @staticmethod
    def normalize_chain_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def strip_command_prefix(self, text: str) -> str:
        lowered = text.lower()
        prefixes = (
            "имитируй",
            "ответь",
            "расскажи",
            "скажи",
            "напиши",
            "сформулируй",
            "подскажи",
            "опиши",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix):
                remainder = text[len(prefix) :].lstrip(" ,:-")
                return remainder or text
        return text

    def remove_descriptor_mentions(
        self,
        text: str,
        descriptor: Optional[str],
        persona_row: sqlite3.Row,
        persona_name: str,
    ) -> str:
        tokens: Set[str] = set()
        if descriptor:
            tokens.add(descriptor)
            plain_descriptor = descriptor.replace("@", "").strip()
            if plain_descriptor:
                tokens.add(plain_descriptor)
        if persona_name:
            tokens.add(persona_name)
            tokens.update(persona_name.split())
        first_name = (persona_row["first_name"] or "").strip()
        if first_name:
            tokens.add(first_name)
        username = persona_row["username"]
        if username:
            tokens.add(username)
            tokens.add(f"@{username}")

        for token in sorted(tokens, key=len, reverse=True):
            if not token:
                continue
            pattern_call = rf"(?i)\bкак\s+{re.escape(token)}\b"
            text = re.sub(pattern_call, "", text)
            pattern_leading = rf"(?i)^\s*{re.escape(token)}([\s,.:;!?\-—]+)"
            text = re.sub(pattern_leading, "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip(" ,:-")

    def clean_imitation_instruction(
        self,
        instruction: Optional[str],
        descriptor: Optional[str],
        persona_row: sqlite3.Row,
        persona_name: str,
    ) -> Optional[str]:
        if not instruction:
            return None
        text = self.strip_call_signs(instruction).strip()
        text = self.strip_command_prefix(text)
        text = self.remove_descriptor_mentions(
            text, descriptor, persona_row, persona_name
        )
        return text or None

    def clean_imitation_payload(
        self,
        payload: Optional[str],
        persona_row: sqlite3.Row,
        persona_name: str,
        descriptor: Optional[str],
    ) -> Optional[str]:
        if not payload:
            return None
        text = self.strip_call_signs(payload).strip()
        text = self.remove_descriptor_mentions(
            text, descriptor, persona_row, persona_name
        )
        return text or None

    def prepare_chain_user_text(
        self,
        *,
        instruction: Optional[str],
        payload: Optional[str],
        descriptor: Optional[str],
        persona_row: sqlite3.Row,
        persona_name: str,
        message: Message,
    ) -> Optional[str]:
        candidates: List[str] = []
        if instruction:
            cleaned_instruction = self.clean_imitation_instruction(
                instruction, descriptor, persona_row, persona_name
            )
            if cleaned_instruction:
                candidates.append(cleaned_instruction)
        if payload:
            cleaned_payload = self.clean_imitation_payload(
                payload, persona_row, persona_name, descriptor
            )
            if cleaned_payload:
                candidates.append(cleaned_payload)
        original = message.text or message.caption or ""
        if original:
            stripped_original = self.strip_call_signs(original)
            stripped_original = self.remove_descriptor_mentions(
                stripped_original, descriptor, persona_row, persona_name
            )
            stripped_original = stripped_original.strip()
            if stripped_original:
                stripped_original = self.strip_command_prefix(stripped_original)
                stripped_original = self.remove_descriptor_mentions(
                    stripped_original, descriptor, persona_row, persona_name
                )
                candidates.append(stripped_original)
        unique: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            normalized = self.normalize_chain_text(candidate)
            if not normalized:
                continue
            if normalized.startswith("/"):
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        if unique:
            return "\n".join(unique)
        return None

    # --------------------------------------------------------------- chain utils ---
    @staticmethod
    def should_skip_chain(cleaned_lower: str) -> bool:
        control_tokens = (
            "имитируй",
            "ответь как",
            "что бы сказал",
            "переведи",
            "translate",
            "перескажи",
            "резюмируй",
            "кратко",
            "перефраз",
            "формулир",
            "исправь",
            "ошиб",
            "список",
            "задач",
        )
        return any(token in cleaned_lower for token in control_tokens)

    def collect_initial_context(
        self, message: Message, *, max_depth: int = 6
    ) -> List[ChainMessage]:
        context: List[ChainMessage] = []
        current = message.reply_to_message
        depth = 0
        trail: List[ChainMessage] = []
        while current is not None and depth < max_depth:
            text = current.text or current.caption or ""
            text = text.strip()
            if text:
                normalized = self.normalize_chain_text(text)
                if normalized:
                    from_user = current.from_user
                    speaker = "Участник"
                    if from_user is not None:
                        speaker = display_name(
                            from_user.username,
                            from_user.first_name,
                            from_user.last_name,
                        )
                    trail.append(
                        ChainMessage(
                            speaker=speaker,
                            text=normalized,
                            is_persona=False,
                        )
                    )
            depth += 1
            current = current.reply_to_message
        context.extend(reversed(trail))
        return context

    def create_chain(
        self,
        chat_id: int,
        persona_row: sqlite3.Row,
        requester: Optional[User],
        user_text: str,
        *,
        context_messages: Optional[List[ChainMessage]] = None,
    ) -> ImitationChain:
        persona_name = display_name(
            persona_row["username"],
            persona_row["first_name"],
            persona_row["last_name"],
        )
        chain = ImitationChain(
            chat_id=chat_id,
            persona_id=int(persona_row["id"]),
            persona_username=persona_row["username"],
            persona_first_name=persona_row["first_name"],
            persona_last_name=persona_row["last_name"],
            persona_name=persona_name,
            messages=[],
        )
        if context_messages:
            for entry in context_messages:
                normalized = self.normalize_chain_text(entry.text)
                if not normalized:
                    continue
                chain.messages.append(
                    ChainMessage(
                        speaker=entry.speaker,
                        text=normalized,
                        is_persona=entry.is_persona,
                    )
                )
        requester_name = "Собеседник"
        if requester is not None:
            requester_name = display_name(
                requester.username, requester.first_name, requester.last_name
            )
        normalized = self.normalize_chain_text(user_text)
        if normalized:
            chain.messages.append(
                ChainMessage(
                    speaker=requester_name,
                    text=normalized,
                    is_persona=False,
                )
            )
        return chain

    def branch_chain(
        self,
        base_chain: ImitationChain,
        requester: Optional[User],
        user_text: str,
    ) -> ImitationChain:
        chain = self._clone_chain(base_chain)
        requester_name = "Собеседник"
        if requester is not None:
            requester_name = display_name(
                requester.username, requester.first_name, requester.last_name
            )
        normalized = self.normalize_chain_text(user_text)
        if normalized:
            chain.messages.append(
                ChainMessage(
                    speaker=requester_name,
                    text=normalized,
                    is_persona=False,
                )
            )
        return chain

    def format_chain_prompt(self, chain: ImitationChain) -> str:
        if not chain.messages:
            return "Ответь в стиле выбранного пользователя."
        history_lines: List[str] = []
        for entry in chain.messages[:-1]:
            speaker = chain.persona_name if entry.is_persona else entry.speaker
            if entry.is_persona:
                speaker = f"{chain.persona_name} (ты)"
            history_lines.append(f"{speaker}: {entry.text}")
        current = chain.messages[-1]
        continuation_hint = " и продолжи цепочку" if history_lines else ""
        current_line = (
            f"{current.speaker} пишет тебе (боту, имитирующему человека): \"{current.text}\"."
            f" Ответь в стиле {chain.persona_name} от первого лица{continuation_hint}."
        )
        if history_lines:
            return "Контекст цепочки:\n" + "\n".join(history_lines) + "\n\n" + current_line
        return current_line

    # --------------------------------------------------------------- descriptor ---
    @staticmethod
    def extract_leading_descriptor(instruction: str) -> Tuple[Optional[str], Optional[str]]:
        text = instruction.strip()
        if not text:
            return None, None
        text = text.lstrip(",:;—- ")
        if not text:
            return None, None
        match = re.match(r"^(@?[\w\-]+)", text)
        if not match:
            return None, None
        descriptor = match.group(1)
        remainder = text[match.end() :].lstrip(" ,:;—-")
        return descriptor, (remainder or None)

    @staticmethod
    def split_imitation_remainder(
        remainder: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        if not remainder:
            return None, None
        remainder = remainder.strip()
        if not remainder:
            return None, None
        if remainder.startswith("@"):
            parts = remainder.split(None, 1)
            descriptor = parts[0]
            payload = parts[1].strip() if len(parts) > 1 else None
            return descriptor, payload
        for delimiter in [":", "—", "–", "-", "\n", ","]:
            idx = remainder.find(delimiter)
            if idx != -1:
                descriptor = remainder[:idx].strip()
                payload = remainder[idx + 1 :].strip() or None
                if descriptor:
                    return descriptor, payload
        remainder = remainder.strip()
        if not remainder:
            return None, None
        parts = remainder.split(None, 1)
        descriptor = parts[0].strip(".,!?-—:;")
        payload = parts[1].strip() if len(parts) > 1 else None
        return descriptor or None, payload

    @staticmethod
    def descriptor_from_prefix(prefix: str) -> Optional[str]:
        prefix = prefix.strip()
        if not prefix:
            return None
        if "@" in prefix:
            match = re.findall(r"@([\w_]{3,})", prefix)
            if match:
                return "@" + match[-1]
        sentences = re.split(r"[.!?]\s*", prefix)
        tail = sentences[-1].strip() if sentences else prefix
        fragments = re.split(r"[:;,\-–—\n]", tail)
        descriptor = fragments[-1].strip() if fragments else tail
        return descriptor or None
