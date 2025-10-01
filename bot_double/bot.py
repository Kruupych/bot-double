from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, TypeVar

from telegram import Message, MessageEntity, Update, User
from telegram.constants import ParseMode
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Settings
from .db import Database
from .style_engine import (
    ContextMessage,
    DialogueParticipant,
    ParticipantProfile,
    RequesterProfile,
    StyleEngine,
    StyleSample,
)
from .relationship_analysis import (
    RelationshipStats,
    build_relationship_hint,
    evaluate_interaction,
)
from .social_analysis import InteractionExcerpt, SocialAnalyzer
from .persona_analysis import PersonaAnalyzer, PersonaSample
from .style_analysis import build_style_summary
from .transcription import SpeechTranscriber
from .utils import (
    display_name,
    guess_gender,
    is_bufferable_message,
    should_store_context_snippet,
    should_store_message,
)

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class _ChatEvent:
    timestamp: int
    user_telegram_id: int
    reply_to_telegram_id: Optional[int]


@dataclass
class _BurstState:
    chat_id: int
    user_id: int
    user_telegram_id: int
    texts: List[str]
    start_timestamp: int
    last_timestamp: int
    last_message: Message
    total_chars: int
    task: Optional[asyncio.Task] = None


class BotDouble:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = Database(settings.db_path, settings.max_messages_per_user)
        self._style = StyleEngine(
            settings.openai_api_key,
            model=settings.openai_model,
            reasoning_effort=settings.openai_reasoning_effort,
            text_verbosity=settings.openai_text_verbosity,
        )
        analysis_model = settings.relationship_analysis_model or settings.openai_model
        self._social = SocialAnalyzer(
            settings.openai_api_key,
            analysis_model,
            reasoning_effort=settings.openai_reasoning_effort,
        )
        persona_model = settings.persona_analysis_model
        self._persona_analyzer = (
            PersonaAnalyzer(
                settings.openai_api_key,
                persona_model,
                reasoning_effort=settings.openai_reasoning_effort,
            )
            if persona_model
            else None
        )
        self._bot_id: Optional[int] = None
        self._bot_name: str = "Бот-Двойник"
        self._bot_user_id: Optional[int] = None
        self._burst_states: Dict[Tuple[int, int], "_BurstState"] = {}
        self._buffer_lock = asyncio.Lock()
        self._chat_events: Dict[int, Deque[_ChatEvent]] = {}
        self._chat_events_maxlen = 50
        self._analysis_queue: Optional[asyncio.Queue[Tuple[int, int, int]]] = None
        self._analysis_inflight: Set[Tuple[int, int, int]] = set()
        self._analysis_worker: Optional[asyncio.Task] = None
        self._persona_queue: Optional[asyncio.Queue[Tuple[int, int]]] = None
        self._persona_inflight: Set[Tuple[int, int]] = set()
        self._persona_worker: Optional[asyncio.Task] = None
        self._transcriber: Optional[SpeechTranscriber] = None
        if self._settings.enable_voice_transcription:
            self._transcriber = SpeechTranscriber(
                settings.openai_api_key,
                settings.voice_transcription_model,
                language=settings.voice_transcription_language,
            )

    def build_application(self) -> Application:
        return (
            ApplicationBuilder()
            .token(self._settings.bot_token)
            .rate_limiter(AIORateLimiter())
            .post_init(self._post_init)
            .post_shutdown(self._post_shutdown)
            .build()
        )

    async def _post_init(self, application: Application) -> None:
        me = await application.bot.get_me()
        self._bot_id = me.id
        self._bot_name = me.first_name or self._bot_name
        try:
            self._bot_user_id = await self._run_db(
                self._db.upsert_user,
                me.id,
                me.username,
                me.first_name,
                me.last_name,
            )
        except Exception:
            LOGGER.exception("Failed to ensure bot profile in database")
        LOGGER.info("Bot initialized as %s (%s)", me.first_name, me.username)
        if self._analysis_queue is None:
            self._analysis_queue = asyncio.Queue()
        if self._analysis_worker is None:
            self._analysis_worker = asyncio.create_task(
                self._relationship_analysis_worker()
            )
        if self._persona_analyzer is not None and self._persona_queue is None:
            self._persona_queue = asyncio.Queue()
        if self._persona_analyzer is not None and self._persona_worker is None:
            self._persona_worker = asyncio.create_task(self._persona_analysis_worker())

    async def _post_shutdown(self, application: Application) -> None:
        LOGGER.info("Bot shutting down")
        await self._flush_all_buffers()
        await self._stop_analysis_worker()
        await self._stop_persona_worker()
        self._db.close()

    # --- handlers ------------------------------------------------------------------
    async def imitate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return
        chat = update.effective_chat
        if chat is None:
            return
        if not context.args:
            await message.reply_text("Использование: /imitate @username Текст-затравка")
            return

        username_arg = context.args[0]
        if not username_arg.startswith("@"):
            await message.reply_text("Первым аргументом должно быть @username")
            return

        username = username_arg.lstrip("@")
        starter = " ".join(context.args[1:]).strip()
        if not starter:
            starter = "Просто продолжи разговор."  # fallback to keep prompt meaningful

        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
            return

        message_count = await self._run_db(
            self._db.get_message_count, chat.id, int(user_row["id"])
        )
        if message_count < self._settings.min_messages_for_profile:
            await message.reply_text(
                f"Пока не могу имитировать @{username}, мне нужно больше его сообщений для анализа"
            )
            return

        samples = await self._collect_style_samples(
            chat.id, int(user_row["id"]), topic_hint=starter
        )
        if not samples:
            await message.reply_text(
                f"Пока не могу имитировать @{username}, сообщений недостаточно."
            )
            return

        persona_card = await self._get_persona_card(chat.id, int(user_row["id"]))
        style_summary = None if persona_card else build_style_summary(samples)

        persona_name = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        context_messages = await self._get_dialog_context(chat.id)
        peer_profiles = await self._collect_peer_profiles(chat.id, int(user_row["id"]))
        requester_profile = await self._collect_requester_profile(
            chat.id, message.from_user, int(user_row["id"])
        )
        persona_gender = guess_gender(
            user_row["first_name"], user_row["username"]
        )
        addressee_internal_id = await self._ensure_internal_user(message.from_user)
        addressee_name = display_name(
            message.from_user.username,
            message.from_user.first_name,
            message.from_user.last_name,
        )
        relationship_hint = await self._relationship_hint_for_addressee(
            chat.id,
            int(user_row["id"]),
            addressee_internal_id,
            addressee_name,
        )
        try:
            ai_reply = await self._generate_reply(
                username,
            persona_name,
            samples,
            starter,
            context_messages,
            peer_profiles,
            requester_profile,
            persona_gender,
            style_summary,
            persona_card,
            relationship_hint,
        )
        except Exception as exc:  # pragma: no cover - network errors etc.
            LOGGER.exception("Failed to generate imitation", exc_info=exc)
            await message.reply_text("Не удалось получить ответ от модели, попробуйте позже")
            return

        await message.reply_text(ai_reply)

    async def imitate_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return

        lines: List[str] = ["Статус профилей:"]
        has_profiles = False
        profiles = await self._run_db(self._db.get_profiles, chat.id)
        for row in profiles:
            has_profiles = True
            persona_name = display_name(
                row["username"], row["first_name"], row["last_name"]
            )
            count = int(row["message_count"])
            if count >= self._settings.min_messages_for_profile:
                marker = "✅"
                info = f"{persona_name} (проанализировано {count} сообщений)"
            else:
                marker = "⏳"
                info = (
                    f"{persona_name} (собрано {count}/{self._settings.min_messages_for_profile} сообщений,"
                    " анализ скоро будет доступен)"
                )
            lines.append(f"{marker} {info}")

        if not has_profiles:
            lines.append("Данных пока нет")

        await message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def auto_imitate_toggle(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *, enabled: bool) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return
        await self._run_db(self._db.set_auto_imitate, chat.id, enabled)
        status = "включена" if enabled else "выключена"
        await message.reply_text(f"Автоимитация {status} для этого чата")

    async def auto_imitate_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.auto_imitate_toggle(update, context, enabled=True)

    async def auto_imitate_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.auto_imitate_toggle(update, context, enabled=False)

    async def on_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None or message.from_user is None:
            return

        # Persist message for style analysis
        await self._capture_message(message)

        is_enabled = await self._run_db(self._db.is_auto_imitate_enabled, chat.id)
        if not is_enabled:
            return

        mention_usernames = self._extract_mentions(message)
        if not mention_usernames:
            return

        username = self._pick_candidate_username(mention_usernames, message.from_user.username)
        if not username:
            return

        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            return

        message_count = await self._run_db(
            self._db.get_message_count, chat.id, int(user_row["id"])
        )
        if message_count < self._settings.min_messages_for_profile:
            return

        if random.random() > self._settings.auto_imitate_probability:
            return

        # Use message text without mentions as starter
        starter = self._strip_mentions(message.text or "", mention_usernames)
        samples = await self._collect_style_samples(
            chat.id,
            int(user_row["id"]),
            topic_hint=starter,
        )
        if not samples:
            return

        persona_card = await self._get_persona_card(chat.id, int(user_row["id"]))
        style_summary = None if persona_card else build_style_summary(samples)

        persona_name = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        context_messages = await self._get_dialog_context(chat.id)
        peer_profiles = await self._collect_peer_profiles(chat.id, int(user_row["id"]))
        requester_profile = await self._collect_requester_profile(
            chat.id, message.from_user, int(user_row["id"])
        )
        persona_gender = guess_gender(
            user_row["first_name"], user_row["username"]
        )
        addressee_internal_id = await self._ensure_internal_user(message.from_user)
        addressee_name = display_name(
            message.from_user.username,
            message.from_user.first_name,
            message.from_user.last_name,
        )
        relationship_hint = await self._relationship_hint_for_addressee(
            chat.id,
            int(user_row["id"]),
            addressee_internal_id,
            addressee_name,
        )
        try:
            ai_reply = await self._generate_reply(
                username,
                persona_name,
                samples,
                starter,
                context_messages,
                peer_profiles,
                requester_profile,
                persona_gender,
                style_summary,
                persona_card,
                relationship_hint,
            )
        except Exception as exc:  # pragma: no cover - network errors etc.
            LOGGER.exception("Auto imitation failed", exc_info=exc)
            return

        await message.reply_text(ai_reply)

    async def on_new_members(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat or not message.new_chat_members:
            return
        if self._bot_id is None:
            self._bot_id = context.bot.id

        for member in message.new_chat_members:
            if member.id == self._bot_id:
                await chat.send_message(self._build_intro_message())
                break

    async def dialogue(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return
        if len(context.args) < 2:
            await message.reply_text(
                "Использование: /dialogue @user1 @user2 [тема диалога]"
            )
            return

        username_a_arg, username_b_arg = context.args[0], context.args[1]
        if not username_a_arg.startswith("@") or not username_b_arg.startswith("@"):
            await message.reply_text("Первых два аргумента должны быть @username участников")
            return

        username_a = username_a_arg.lstrip("@")
        username_b = username_b_arg.lstrip("@")
        topic = " ".join(context.args[2:]).strip()

        row_a = await self._run_db(self._db.get_user_by_username, username_a)
        row_b = await self._run_db(self._db.get_user_by_username, username_b)
        if row_a is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username_a}.")
            return
        if row_b is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username_b}.")
            return

        internal_a = int(row_a["id"])
        internal_b = int(row_b["id"])

        count_a = await self._run_db(self._db.get_message_count, chat.id, internal_a)
        count_b = await self._run_db(self._db.get_message_count, chat.id, internal_b)
        threshold = self._settings.min_messages_for_profile
        if count_a < threshold:
            await message.reply_text(
                f"Пока не могу построить диалог: @{username_a} имеет только {count_a}/{threshold} сообщений."
            )
            return
        if count_b < threshold:
            await message.reply_text(
                f"Пока не могу построить диалог: @{username_b} имеет только {count_b}/{threshold} сообщений."
            )
            return

        samples_a = await self._collect_style_samples(
            chat.id,
            internal_a,
            topic_hint=topic,
        )
        samples_b = await self._collect_style_samples(
            chat.id,
            internal_b,
            topic_hint=topic,
        )
        if not samples_a or not samples_b:
            await message.reply_text("Недостаточно примеров стиля для построения диалога.")
            return

        persona_name_a = display_name(
            row_a["username"], row_a["first_name"], row_a["last_name"]
        )
        persona_name_b = display_name(
            row_b["username"], row_b["first_name"], row_b["last_name"]
        )

        persona_card_a = await self._get_persona_card(chat.id, internal_a)
        persona_card_b = await self._get_persona_card(chat.id, internal_b)
        style_summary_a = None if persona_card_a else build_style_summary(samples_a)
        style_summary_b = None if persona_card_b else build_style_summary(samples_b)

        relationship_hint_a = await self._relationship_hint_for_addressee(
            chat.id,
            internal_a,
            internal_b,
            persona_name_b,
        )
        relationship_hint_b = await self._relationship_hint_for_addressee(
            chat.id,
            internal_b,
            internal_a,
            persona_name_a,
        )

        try:
            dialogue_text = await self._generate_dialogue(
                username_a,
                persona_name_a,
                samples_a,
                style_summary_a,
                persona_card_a,
                relationship_hint_a,
                username_b,
                persona_name_b,
                samples_b,
                style_summary_b,
                persona_card_b,
                relationship_hint_b,
                topic or ""
            )
        except Exception as exc:  # pragma: no cover - network errors etc.
            LOGGER.exception("Failed to generate dialogue", exc_info=exc)
            await message.reply_text("Не удалось построить диалог, попробуйте позже")
            return

        await message.reply_text(dialogue_text)

    async def profile_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat or message.from_user is None:
            return

        requester = message.from_user
        requester_internal_id = await self._ensure_internal_user(requester)
        if requester_internal_id is None:
            await message.reply_text("Не удалось определить ваш профиль.")
            return

        args = context.args
        target_internal_id = requester_internal_id
        target_row = await self._run_db(self._db.get_user_by_id, requester_internal_id)
        relationship_target_internal_id: Optional[int] = None

        if args:
            first = args[0]
            if not first.startswith("@"):
                await message.reply_text("Использование: /profile [@user] [@other]")
                return
            username = first.lstrip("@")
            user_row = await self._run_db(self._db.get_user_by_username, username)
            if user_row is None:
                await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
                return
            target_internal_id = int(user_row["id"])
            target_row = await self._run_db(self._db.get_user_by_id, target_internal_id)
            if target_row is None:
                await message.reply_text("Нет данных о выбранном пользователе.")
                return
            if len(args) >= 2:
                second = args[1]
                if not second.startswith("@"):
                    await message.reply_text("Второй аргумент должен быть в формате @username")
                    return
                second_username = second.lstrip("@")
                second_row = await self._run_db(
                    self._db.get_user_by_username, second_username
                )
                if second_row is None:
                    await message.reply_text(
                        f"Я ещё не знаю пользователя @{second_username}."
                    )
                    return
                relationship_target_internal_id = int(second_row["id"])

        target_name = display_name(
            target_row["username"], target_row["first_name"], target_row["last_name"]
        )

        persona_card = await self._get_persona_card(chat.id, target_internal_id)
        response_lines: List[str] = [f"Профиль {target_name}:"]
        if persona_card:
            response_lines.append(persona_card)
        else:
            response_lines.append("Карточка персоны ещё не готова. Продолжайте общаться!")

        relationship_lines: List[str] = []
        if relationship_target_internal_id is not None:
            summary = await self._get_relationship_summary_text(
                chat.id, target_internal_id, relationship_target_internal_id
            )
            other_row = await self._run_db(
                self._db.get_user_by_id, relationship_target_internal_id
            )
            other_name = display_name(
                other_row["username"], other_row["first_name"], other_row["last_name"]
            ) if other_row else "неизвестный пользователь"
            if summary:
                relationship_lines.append(
                    f"Отношение к {other_name}: {summary}"
                )
            else:
                relationship_lines.append(
                    f"Отношение к {other_name}: данных пока мало."
                )
            reverse = await self._get_relationship_summary_text(
                chat.id, relationship_target_internal_id, target_internal_id
            )
            if reverse:
                relationship_lines.append(
                    f"Ответная позиция {other_name}: {reverse}"
                )
        elif target_internal_id != requester_internal_id:
            summary = await self._get_relationship_summary_text(
                chat.id, requester_internal_id, target_internal_id
            )
            if summary:
                response_lines.append("")
                response_lines.append(
                    f"Вы о {target_name}: {summary}"
                )
            reverse = await self._get_relationship_summary_text(
                chat.id, target_internal_id, requester_internal_id
            )
            if reverse:
                response_lines.append(
                    f"{target_name} о вас: {reverse}"
                )
        elif relationship_target_internal_id is None:
            # requester looking at self; optionally show relationships for provided second mention later
            pass

        if relationship_lines:
            response_lines.append("")
            response_lines.extend(relationship_lines)

        text = "\n".join(line for line in response_lines if line is not None)
        await message.reply_text(text or "Нет данных", disable_web_page_preview=True)

    # --- internal helpers -----------------------------------------------------------
    async def _capture_message(self, message: Message) -> None:
        if message.from_user is None:
            return
        user = message.from_user
        if user.is_bot:
            if self._bot_id is None or user.id != self._bot_id:
                return
        if message.via_bot is not None:
            return
        if message.forward_origin is not None:
            return
        chat_id = message.chat_id
        if chat_id is None:
            return
        user_id = await self._run_db(
            self._db.upsert_user,
            user.id,
            user.username,
            user.first_name,
            user.last_name,
        )
        if user.is_bot and user.id == self._bot_id:
            self._bot_user_id = user_id
        timestamp = int(message.date.timestamp())
        key = (chat_id, user_id)

        text_source_voice = False
        text = message.text or ""
        if not text and self._settings.enable_voice_transcription:
            text = await self._maybe_transcribe_voice_message(message)
            if text:
                text_source_voice = True
        if not text:
            return
        text = text.strip()
        if not text:
            return

        if text_source_voice:
            if text.startswith(('/', '!', '.')):
                await self._flush_buffer_for_key(key)
                self._record_chat_event(message, timestamp)
                return
            should_store = should_store_context_snippet(
                text, min_tokens=self._settings.min_tokens_to_store
            )
            lowered = text.lower()
            bufferable = (
                not should_store
                and "http://" not in lowered
                and "https://" not in lowered
            )
        else:
            should_store = should_store_message(
                message,
                min_tokens=self._settings.min_tokens_to_store,
                allowed_bot_id=self._bot_id,
            )
            bufferable = is_bufferable_message(
                message,
                allowed_bot_id=self._bot_id,
            )

        if should_store or bufferable:
            await self._append_to_burst(
                key,
                message,
                user_id,
                user.id,
                text,
                timestamp,
            )
        else:
            await self._flush_buffer_for_key(key)

        context_snippet = self._extract_command_context(text)
        if context_snippet and should_store_context_snippet(
            context_snippet, min_tokens=self._settings.min_tokens_to_store
        ):
            await self._run_db(
                self._db.store_message,
                message.chat_id,
                user_id,
                context_snippet,
                timestamp,
                context_only=True,
            )

        self._record_chat_event(message, timestamp)

    def _extract_command_context(self, text: str) -> Optional[str]:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped.split(maxsplit=1)
        if not parts:
            return None
        command_token = parts[0]
        command = command_token.split("@", maxsplit=1)[0].lower()
        if command != "/imitate":
            return None
        if len(parts) < 2:
            return None
        remainder = parts[1].strip()
        return remainder or None

    async def _update_pair_interactions(
        self, message: Message, speaker_internal_id: int, text: str
    ) -> None:
        chat_id = message.chat_id
        if chat_id is None:
            return
        stripped = text.strip()
        if not stripped or stripped.startswith("/") or len(stripped) < 3:
            return

        targets: Set[int] = set()

        mention_usernames = self._extract_mentions(message)
        for username in mention_usernames:
            row = await self._run_db(self._db.get_user_by_username, username)
            if row is None:
                continue
            targets.add(int(row["id"]))

        reply = message.reply_to_message
        if reply and reply.from_user:
            reply_user = reply.from_user
            target_id = await self._run_db(
                self._db.upsert_user,
                reply_user.id,
                reply_user.username,
                reply_user.first_name,
                reply_user.last_name,
            )
            targets.add(target_id)

        if not targets:
            return

        signals = evaluate_interaction(stripped)
        sample_text = stripped if signals.has_any() else None
        timestamp = int(message.date.timestamp())

        for target_id in targets:
            if target_id == speaker_internal_id:
                continue
            await self._run_db(
                self._db.update_pair_stats,
                chat_id,
                speaker_internal_id,
                target_id,
                informal=signals.informal,
                formal=signals.formal,
                teasing=signals.teasing,
                sample_text=sample_text,
                full_text=stripped,
                timestamp=timestamp,
            )
            await self._maybe_queue_relationship_analysis(
                chat_id, speaker_internal_id, target_id
            )

    def _extract_mentions(self, message: Message) -> List[str]:
        entities = message.parse_entities([MessageEntity.MENTION])
        usernames = []
        for entity, value in entities.items():
            username = value.lstrip("@")
            if username:
                usernames.append(username)
        return usernames

    def _pick_candidate_username(
        self, usernames: List[str], author_username: Optional[str]
    ) -> Optional[str]:
        for username in usernames:
            if author_username and username.lower() == author_username.lower():
                continue
            return username
        return None

    def _strip_mentions(self, text: str, usernames: List[str]) -> str:
        cleaned = text
        for username in usernames:
            cleaned = cleaned.replace(f"@{username}", "").strip()
        return cleaned or "Продолжи диалог."

    async def _ensure_internal_user(self, user: Optional[User]) -> Optional[int]:
        if user is None:
            return None
        return await self._run_db(
            self._db.upsert_user,
            user.id,
            user.username,
            user.first_name,
            user.last_name,
        )

    async def _relationship_hint_for_addressee(
        self,
        chat_id: int,
        speaker_internal_id: int,
        target_internal_id: Optional[int],
        addressee_name: str,
    ) -> Optional[str]:
        if target_internal_id is None:
            return None
        stats = await self._run_db(
            self._db.get_pair_stats,
            chat_id,
            speaker_internal_id,
            target_internal_id,
        )
        if not stats:
            return None
        relationship_stats = RelationshipStats(
            total=int(stats["total_count"]),
            informal=int(stats["informal_count"]),
            formal=int(stats["formal_count"]),
            teasing=int(stats["teasing_count"]),
            samples=list(stats.get("samples", [])),
            summary=stats.get("analysis_summary"),
        )
        return build_relationship_hint(addressee_name, relationship_stats)

    async def _get_persona_card(self, chat_id: int, user_id: int) -> Optional[str]:
        profile = await self._run_db(
            self._db.get_persona_profile,
            chat_id,
            user_id,
        )
        if not profile:
            return None
        lines: List[str] = []
        summary = profile.get("summary")
        if summary:
            summary = str(summary).strip()
            if summary:
                lines.append(summary)
        details = profile.get("details")
        if details:
            try:
                data = json.loads(details)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                interests = data.get("interests")
                if isinstance(interests, list) and interests:
                    joined = ", ".join(str(item) for item in interests if item)
                    if joined:
                        lines.append(f"Интересы: {joined}")
                humor = str(data.get("humor_style", "")).strip()
                if humor and humor.lower() != "unknown":
                    lines.append(f"Чувство юмора: {humor}")
                emotionality = str(data.get("emotionality", "")).strip()
                if emotionality and emotionality.lower() != "unknown":
                    lines.append(f"Эмоциональность: {emotionality}")
                tonality = str(data.get("tonality", "")).strip()
                if tonality and tonality.lower() != "unknown":
                    lines.append(f"Тональность: {tonality}")
                speech_traits = data.get("speech_traits")
                if isinstance(speech_traits, list) and speech_traits:
                    joined = ", ".join(str(item) for item in speech_traits if item)
                    if joined:
                        lines.append(f"Речевые привычки: {joined}")
                tips = str(data.get("writing_tips", "")).strip()
                if tips and tips.lower() != "unknown":
                    lines.append(f"Советы по стилю: {tips}")
        persona_card = "\n".join(lines).strip()
        return persona_card or None

    async def _get_relationship_summary_text(
        self, chat_id: int, speaker_id: int, target_id: int
    ) -> Optional[str]:
        stats = await self._run_db(
            self._db.get_pair_stats,
            chat_id,
            speaker_id,
            target_id,
        )
        if not stats:
            return None
        summary = stats.get("analysis_summary")
        if summary and str(summary).strip():
            return str(summary).strip()
        target_row = await self._run_db(self._db.get_user_by_id, target_id)
        if target_row is None:
            return None
        target_name = display_name(
            target_row["username"], target_row["first_name"], target_row["last_name"]
        )
        relationship_stats = RelationshipStats(
            total=int(stats.get("total_count", 0)),
            informal=int(stats.get("informal_count", 0)),
            formal=int(stats.get("formal_count", 0)),
            teasing=int(stats.get("teasing_count", 0)),
            samples=list(stats.get("samples", [])),
        )
        return build_relationship_hint(target_name, relationship_stats)

    async def _note_persona_message(
        self, chat_id: Optional[int], user_id: int, timestamp: int
    ) -> None:
        if chat_id is None:
            return
        if self._persona_analyzer is None or self._persona_queue is None:
            return
        if self._bot_user_id is not None and user_id == self._bot_user_id:
            return
        await self._run_db(self._db.increment_persona_pending, chat_id, user_id)
        await self._maybe_queue_persona_analysis(chat_id, user_id)

    async def _maybe_queue_relationship_analysis(
        self, chat_id: int, speaker_id: int, target_id: int
    ) -> None:
        if self._analysis_queue is None:
            return
        key = (chat_id, speaker_id, target_id)
        if key in self._analysis_inflight:
            return
        stats = await self._run_db(
            self._db.get_pair_stats,
            chat_id,
            speaker_id,
            target_id,
        )
        if not stats:
            return
        pending = int(stats.get("pending_messages", 0))
        if pending < self._settings.relationship_analysis_min_pending:
            return
        last_analyzed = stats.get("last_analyzed_at")
        threshold_seconds = self._settings.relationship_analysis_min_hours * 3600
        if (
            threshold_seconds > 0
            and last_analyzed is not None
            and int(time.time()) - int(last_analyzed) < threshold_seconds
        ):
            return
        await self._analysis_queue.put(key)
        self._analysis_inflight.add(key)

    async def _maybe_queue_persona_analysis(self, chat_id: int, user_id: int) -> None:
        if self._persona_queue is None or self._persona_analyzer is None:
            return
        key = (chat_id, user_id)
        if key in self._persona_inflight:
            return
        profile = await self._run_db(
            self._db.get_persona_profile,
            chat_id,
            user_id,
        )
        if not profile:
            return
        pending = int(profile.get("pending_messages", 0))
        if pending < self._settings.persona_analysis_min_messages:
            return
        last_analyzed = profile.get("last_analyzed_at")
        threshold_seconds = self._settings.persona_analysis_min_hours * 3600
        if (
            threshold_seconds > 0
            and last_analyzed is not None
            and int(time.time()) - int(last_analyzed) < threshold_seconds
        ):
            return
        await self._persona_queue.put(key)
        self._persona_inflight.add(key)

    async def _append_to_burst(
        self,
        key: Tuple[int, int],
        message: Message,
        user_internal_id: int,
        user_telegram_id: int,
        text: str,
        timestamp: int,
    ) -> None:
        normalized = text.strip()
        if not normalized:
            await self._flush_buffer_for_key(key)
            return

        chat_id = message.chat_id
        if chat_id is None:
            return

        buffer_to_flush: Optional[_BurstState] = None
        flush_now = False
        async with self._buffer_lock:
            burst = self._burst_states.get(key)
            if burst:
                if self._should_break_burst(
                    burst, message, user_telegram_id, timestamp
                ):
                    buffer_to_flush = self._burst_states.pop(key)
                    if burst.task:
                        burst.task.cancel()
                    burst = None
            if burst is None:
                burst = _BurstState(
                    chat_id=chat_id,
                    user_id=user_internal_id,
                    user_telegram_id=user_telegram_id,
                    texts=[normalized],
                    start_timestamp=timestamp,
                    last_timestamp=timestamp,
                    last_message=message,
                    total_chars=len(normalized),
                )
                self._burst_states[key] = burst
            else:
                burst.texts.append(normalized)
                burst.last_timestamp = timestamp
                burst.last_message = message
                burst.total_chars += len(normalized)
            if burst.task:
                burst.task.cancel()
            burst.task = asyncio.create_task(self._delayed_flush(key))
            flush_now = self._burst_limits_exceeded(burst)

        if buffer_to_flush is not None:
            await self._store_burst(buffer_to_flush)

        if flush_now:
            await self._flush_buffer_for_key(key)

    async def _delayed_flush(self, key: Tuple[int, int]) -> None:
        try:
            await asyncio.sleep(self._settings.burst_inactivity_seconds)
            await self._flush_buffer_for_key(key)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
            return

    async def _flush_buffer_for_key(self, key: Tuple[int, int]) -> None:
        burst = await self._pop_burst_for_key(key)
        if burst is None:
            return
        await self._store_burst(burst)

    async def _pop_burst_for_key(self, key: Tuple[int, int]) -> Optional[_BurstState]:
        async with self._buffer_lock:
            burst = self._burst_states.get(key)
            if not burst or not burst.texts:
                if burst and burst.task:
                    burst.task.cancel()
                    burst.task = None
                if burst:
                    del self._burst_states[key]
                return None
            if burst.task:
                burst.task.cancel()
                burst.task = None
            del self._burst_states[key]
            return burst

    async def _store_burst(self, burst: _BurstState) -> None:
        combined_text = " ".join(burst.texts).strip()
        if not combined_text:
            return
        if not should_store_context_snippet(
            combined_text, min_tokens=self._settings.min_tokens_to_store
        ):
            return

        await self._run_db(
            self._db.store_message,
            burst.chat_id,
            burst.user_id,
            combined_text,
            burst.last_timestamp,
            context_only=False,
        )
        await self._update_pair_interactions(
            burst.last_message, burst.user_id, combined_text
        )
        await self._note_persona_message(
            burst.chat_id, burst.user_id, burst.last_timestamp
        )

    def _should_break_burst(
        self,
        burst: _BurstState,
        message: Message,
        user_telegram_id: int,
        timestamp: int,
    ) -> bool:
        if timestamp - burst.last_timestamp > self._settings.burst_gap_seconds:
            return True
        if self._settings.burst_max_duration_seconds > 0 and (
            timestamp - burst.start_timestamp
        ) > self._settings.burst_max_duration_seconds:
            return True
        reply_to_id = (
            message.reply_to_message.from_user.id
            if message.reply_to_message and message.reply_to_message.from_user
            else None
        )
        if self._was_interrupted_by_turn(
            burst.chat_id,
            burst.user_telegram_id,
            burst.last_timestamp,
            timestamp,
            reply_to_id,
        ):
            return True
        return False

    def _burst_limits_exceeded(self, burst: _BurstState) -> bool:
        if (
            self._settings.burst_max_parts > 0
            and len(burst.texts) >= self._settings.burst_max_parts
        ):
            return True
        if (
            self._settings.burst_max_chars > 0
            and burst.total_chars >= self._settings.burst_max_chars
        ):
            return True
        return False

    def _was_interrupted_by_turn(
        self,
        chat_id: int,
        user_telegram_id: int,
        last_timestamp: int,
        current_timestamp: int,
        reply_to_telegram_id: Optional[int],
    ) -> bool:
        events = self._chat_events.get(chat_id)
        if not events or not self._settings.enable_bursts:
            return False
        window = self._settings.turn_window_seconds
        for event in events:
            if event.timestamp <= last_timestamp:
                continue
            if event.timestamp >= current_timestamp:
                break
            if event.user_telegram_id == user_telegram_id:
                continue
            if reply_to_telegram_id is not None and (
                event.user_telegram_id == reply_to_telegram_id
            ):
                return True
            if event.reply_to_telegram_id == user_telegram_id:
                return True
            if (
                current_timestamp - event.timestamp <= window
                and event.timestamp - last_timestamp <= window
            ):
                return True
        return False

    def _record_chat_event(self, message: Message, timestamp: int) -> None:
        chat_id = message.chat_id
        if chat_id is None or message.from_user is None:
            return
        reply_to_id = (
            message.reply_to_message.from_user.id
            if message.reply_to_message and message.reply_to_message.from_user
            else None
        )
        event = _ChatEvent(
            timestamp=timestamp,
            user_telegram_id=message.from_user.id,
            reply_to_telegram_id=reply_to_id,
        )
        events = self._chat_events.get(chat_id)
        if events is None:
            events = deque(maxlen=self._chat_events_maxlen)
            self._chat_events[chat_id] = events
        events.append(event)

    async def _maybe_transcribe_voice_message(self, message: Message) -> Optional[str]:
        if self._transcriber is None:
            return None
        voice = message.voice
        if voice is None:
            return None
        if (
            self._settings.voice_transcription_max_duration > 0
            and voice.duration > self._settings.voice_transcription_max_duration
        ):
            LOGGER.info(
                "Skipping transcription: voice message too long (%ss)", voice.duration
            )
            return None
        try:
            file = await message.get_file()
            buffer = io.BytesIO()
            await file.download_to_memory(out=buffer)
        except Exception:
            LOGGER.exception("Failed to download voice message for transcription")
            return None

        audio_bytes = buffer.getvalue()
        if not audio_bytes:
            return None
        filename = f"voice_{voice.file_unique_id or voice.file_id}.ogg"
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(
                None,
                partial(
                    self._transcriber.transcribe,
                    audio_bytes,
                    filename,
                    voice.mime_type,
                    language=self._settings.voice_transcription_language,
                ),
            )
        except Exception:
            LOGGER.exception("Failed to transcribe voice message")
            return None
        if text:
            LOGGER.debug("Voice transcription succeeded: %s", text)
        return text

    async def _flush_all_buffers(self) -> None:
        async with self._buffer_lock:
            keys = list(self._burst_states.keys())
        for key in keys:
            await self._flush_buffer_for_key(key)

    async def _generate_reply(
        self,
        username: str,
        persona_name: str,
        samples: List[str],
        starter: str,
        context_messages: Optional[List[ContextMessage]],
        peer_profiles: Optional[List[ParticipantProfile]],
        requester_profile: Optional[RequesterProfile],
        persona_gender: Optional[str],
        style_summary: Optional[str],
        persona_card: Optional[str],
        relationship_hint: Optional[str],
    ) -> str:
        loop = asyncio.get_running_loop()
        style_samples = [StyleSample(text=sample) for sample in samples]
        return await loop.run_in_executor(
            None,
            self._style.generate_reply,
            username,
            persona_name,
            style_samples,
            starter,
            context_messages,
            peer_profiles,
            requester_profile,
            persona_gender,
            style_summary,
            persona_card,
            relationship_hint,
        )

    async def _generate_dialogue(
        self,
        username_a: str,
        persona_name_a: str,
        samples_a: List[str],
        style_summary_a: Optional[str],
        persona_card_a: Optional[str],
        relationship_hint_a: Optional[str],
        username_b: str,
        persona_name_b: str,
        samples_b: List[str],
        style_summary_b: Optional[str],
        persona_card_b: Optional[str],
        relationship_hint_b: Optional[str],
        topic: str,
    ) -> str:
        loop = asyncio.get_running_loop()
        participant_a = DialogueParticipant(
            username=username_a,
            name=persona_name_a,
            samples=[StyleSample(text=sample) for sample in samples_a],
            style_summary=style_summary_a,
            persona_card=persona_card_a,
            relationship_hint=relationship_hint_a,
        )
        participant_b = DialogueParticipant(
            username=username_b,
            name=persona_name_b,
            samples=[StyleSample(text=sample) for sample in samples_b],
            style_summary=style_summary_b,
            persona_card=persona_card_b,
            relationship_hint=relationship_hint_b,
        )
        return await loop.run_in_executor(
            None,
            self._style.generate_dialogue,
            participant_a,
            participant_b,
            topic,
        )

    async def _relationship_analysis_worker(self) -> None:
        assert self._analysis_queue is not None
        try:
            while True:
                key = await self._analysis_queue.get()
                try:
                    await self._perform_relationship_analysis(*key)
                except Exception:
                    LOGGER.exception("Relationship analysis task failed")
                finally:
                    self._analysis_inflight.discard(key)
                    self._analysis_queue.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            if self._analysis_queue is not None:
                while not self._analysis_queue.empty():
                    try:
                        self._analysis_queue.get_nowait()
                        self._analysis_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

    async def _perform_relationship_analysis(
        self, chat_id: int, speaker_id: int, target_id: int
    ) -> None:
        stats = await self._run_db(
            self._db.get_pair_stats,
            chat_id,
            speaker_id,
            target_id,
        )
        if not stats:
            return
        pending = int(stats.get("pending_messages", 0))
        if pending < self._settings.relationship_analysis_min_pending:
            return

        speaker_row = await self._run_db(self._db.get_user_by_id, speaker_id)
        target_row = await self._run_db(self._db.get_user_by_id, target_id)
        if speaker_row is None or target_row is None:
            return

        speaker_name = display_name(
            speaker_row["username"],
            speaker_row["first_name"],
            speaker_row["last_name"],
        )
        target_name = display_name(
            target_row["username"],
            target_row["first_name"],
            target_row["last_name"],
        )

        max_samples = 20
        rows = await self._run_db(
            self._db.get_pair_messages,
            chat_id,
            speaker_id,
            target_id,
            max_samples,
        )
        if not rows:
            return

        min_required = max(
            self._settings.relationship_analysis_min_pending, 5
        )
        if len(rows) < min_required:
            return

        excerpts: List[InteractionExcerpt] = []
        for record in reversed(rows):
            focus_text = (record["text"] or "").strip()
            if not focus_text:
                continue
            focus_timestamp = int(record["timestamp"])
            context_rows = await self._run_db(
                self._db.get_chat_messages_before,
                chat_id,
                focus_timestamp,
                3,
            )
            context_lines: List[str] = []
            for ctx in context_rows[-3:]:
                ctx_text = (ctx["text"] or "").strip()
                if not ctx_text:
                    continue
                ctx_name = display_name(
                    ctx["username"], ctx["first_name"], ctx["last_name"]
                )
                context_lines.append(f"{ctx_name}: {ctx_text}")
            excerpts.append(
                InteractionExcerpt(
                    focus_text=focus_text,
                    focus_timestamp=focus_timestamp,
                    context=tuple(context_lines),
                )
            )

        if len(excerpts) < min_required:
            return

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._social.analyze_relationship(
                speaker_name, target_name, tuple(excerpts)
            ),
        )
        if not result:
            return

        summary = str(result.get("summary", "")).strip()
        if not summary:
            tone = str(result.get("tone", "unknown")).strip()
            notes = str(result.get("emotional_notes", "unknown")).strip()
            summary = f"Тон общения: {tone}. Эмоциональные заметки: {notes}."
        details_json = json.dumps(result, ensure_ascii=False)
        analyzed_at = int(time.time())
        await self._run_db(
            self._db.save_pair_analysis,
            chat_id,
            speaker_id,
            target_id,
            summary,
            details_json,
            analyzed_at,
        )

    async def _persona_analysis_worker(self) -> None:
        assert self._persona_queue is not None
        try:
            while True:
                key = await self._persona_queue.get()
                try:
                    await self._perform_persona_analysis(*key)
                except Exception:
                    LOGGER.exception("Persona analysis task failed")
                finally:
                    self._persona_inflight.discard(key)
                    self._persona_queue.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            if self._persona_queue is not None:
                while not self._persona_queue.empty():
                    try:
                        self._persona_queue.get_nowait()
                        self._persona_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

    async def _perform_persona_analysis(self, chat_id: int, user_id: int) -> None:
        if self._persona_analyzer is None:
            return
        profile = await self._run_db(
            self._db.get_persona_profile,
            chat_id,
            user_id,
        )
        if not profile:
            return
        pending = int(profile.get("pending_messages", 0))
        if pending < self._settings.persona_analysis_min_messages:
            return

        user_row = await self._run_db(self._db.get_user_by_id, user_id)
        if user_row is None:
            return

        messages = await self._run_db(
            self._db.get_recent_messages_with_timestamp,
            chat_id,
            user_id,
            self._settings.persona_analysis_max_messages,
        )
        if not messages:
            return

        samples: List[PersonaSample] = []
        for row in messages:
            text = (row["text"] or "").strip()
            if not text:
                continue
            samples.append(
                PersonaSample(text=text, timestamp=int(row["timestamp"]))
            )

        if len(samples) < self._settings.persona_analysis_min_messages:
            return

        display = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._persona_analyzer.build_persona_card(
                display, tuple(samples)
            ),
        )
        if not result:
            return

        summary = str(result.get("overall_summary", "")).strip()
        if not summary:
            summary = str(result.get("writing_tips", "")).strip()
        if not summary:
            interests = result.get("interests") or []
            if isinstance(interests, list) and interests:
                joined = ", ".join(str(item) for item in interests if item)
                if joined:
                    summary = f"Интересы: {joined}."
        if not summary:
            summary = "Карточка персоны обновлена."

        details_json = json.dumps(result, ensure_ascii=False)
        analyzed_at = int(time.time())
        await self._run_db(
            self._db.save_persona_profile,
            chat_id,
            user_id,
            summary,
            details_json,
            analyzed_at,
        )

    async def _stop_analysis_worker(self) -> None:
        if self._analysis_worker is not None:
            self._analysis_worker.cancel()
            try:
                await self._analysis_worker
            except asyncio.CancelledError:
                pass
            self._analysis_worker = None
        if self._analysis_queue is not None:
            while not self._analysis_queue.empty():
                try:
                    self._analysis_queue.get_nowait()
                    self._analysis_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            self._analysis_queue = None
        self._analysis_inflight.clear()

    async def _stop_persona_worker(self) -> None:
        if self._persona_worker is not None:
            self._persona_worker.cancel()
            try:
                await self._persona_worker
            except asyncio.CancelledError:
                pass
            self._persona_worker = None
        if self._persona_queue is not None:
            while not self._persona_queue.empty():
                try:
                    self._persona_queue.get_nowait()
                    self._persona_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            self._persona_queue = None
        self._persona_inflight.clear()

    async def _collect_style_samples(
        self, chat_id: int, user_id: int, *, topic_hint: Optional[str] = None
    ) -> List[str]:
        total_target = self._settings.prompt_samples
        if total_target <= 0:
            return []

        recent_limit = min(self._settings.style_recent_messages, total_target)
        recent_messages = []
        if recent_limit > 0:
            recent_messages = await self._run_db(
                self._db.get_recent_messages_for_user,
                chat_id,
                user_id,
                recent_limit,
            )

        random_needed = max(total_target - len(recent_messages), 0)
        random_pool: List[str] = []
        if random_needed > 0:
            random_fetch = max(random_needed * 2, total_target)
            random_pool = await self._run_db(
                self._db.get_random_messages,
                chat_id,
                user_id,
                random_fetch,
            )

        tokens = self._prepare_topic_tokens(topic_hint)
        candidates: List[tuple[str, bool, int]] = []
        seen: Set[str] = set()

        order = 0
        for text in recent_messages:
            normalized = text.strip()
            if not normalized or normalized in seen:
                continue
            candidates.append((normalized, True, order))
            seen.add(normalized)
            order += 1

        for text in random_pool:
            normalized = text.strip()
            if not normalized or normalized in seen:
                continue
            candidates.append((normalized, False, order))
            seen.add(normalized)
            order += 1

        if not candidates:
            return []

        if tokens:
            scored = []
            for text, is_recent, idx in candidates:
                score = self._topical_score(text, tokens)
                recency_bonus = 1 if is_recent else 0
                scored.append((text, score, recency_bonus, idx))
            scored.sort(key=lambda item: (-item[1], -item[2], item[3]))
            selected = [item[0] for item in scored[:total_target]]
            return selected

        return [text for text, _, _ in candidates[:total_target]]

    def _prepare_topic_tokens(self, topic_hint: Optional[str]) -> Set[str]:
        if not topic_hint:
            return set()
        tokens = {token for token in re.findall(r"[\w']+", topic_hint.lower()) if len(token) > 2}
        return tokens

    def _topical_score(self, text: str, topic_tokens: Set[str]) -> float:
        if not topic_tokens:
            return 0.0
        words = {token for token in re.findall(r"[\w']+", text.lower()) if token}
        if not words:
            return 0.0
        overlap = topic_tokens & words
        if not overlap:
            return 0.0
        return len(overlap) / len(topic_tokens)

    async def _collect_peer_profiles(
        self, chat_id: int, target_user_id: int
    ) -> Optional[List[ParticipantProfile]]:
        if self._settings.peer_profile_count <= 0 or self._settings.peer_profile_samples <= 0:
            return None

        peers = await self._run_db(
            self._db.get_top_participants,
            chat_id,
            target_user_id,
            self._settings.peer_profile_count,
        )
        if not peers:
            return None

        profiles: List[ParticipantProfile] = []
        for row in peers:
            if self._bot_user_id is not None and int(row["id"]) == self._bot_user_id:
                continue
            peer_samples = await self._run_db(
                self._db.get_random_messages,
                chat_id,
                int(row["id"]),
                self._settings.peer_profile_samples,
            )
            if not peer_samples:
                continue
            name = display_name(row["username"], row["first_name"], row["last_name"])
            formatted = [sample.strip() for sample in peer_samples if sample.strip()]
            if not formatted:
                continue
            profiles.append(ParticipantProfile(name=name, samples=formatted))
        return profiles or None

    async def _collect_requester_profile(
        self, chat_id: int, user: Optional[User], target_internal_id: int
    ) -> Optional[RequesterProfile]:
        if user is None:
            return None
        requester_internal_id = await self._run_db(
            self._db.upsert_user,
            user.id,
            user.username,
            user.first_name,
            user.last_name,
        )
        sample_limit = max(self._settings.style_recent_messages, 3)
        samples = await self._run_db(
            self._db.get_recent_messages_for_user,
            chat_id,
            requester_internal_id,
            sample_limit,
        )
        formatted = [text.strip() for text in samples if text.strip()]
        name = display_name(user.username, user.first_name, user.last_name)
        is_same_person = requester_internal_id == target_internal_id
        if not formatted and not is_same_person:
            return None
        return RequesterProfile(
            name=name,
            samples=formatted[:sample_limit],
            is_same_person=is_same_person,
        )

    async def _run_db(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        bound = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound)

    async def forget_me(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or message.from_user is None:
            return
        deleted = await self._run_db(self._db.delete_user_data, message.from_user.id)
        if deleted:
            await message.reply_text("Ваши сообщения и профиль удалены. Я забуду вас.")
        else:
            await message.reply_text("У меня не было сохранённых данных о вас.")

    async def _get_dialog_context(self, chat_id: int) -> Optional[List[ContextMessage]]:
        if self._settings.dialog_context_messages <= 0:
            return None
        rows = await self._run_db(
            self._db.get_recent_chat_messages,
            chat_id,
            self._settings.dialog_context_messages,
        )
        if not rows:
            return None
        messages: List[ContextMessage] = []
        for row in rows:
            text = (row["text"] or "").strip()
            if not text:
                continue
            persona_name = display_name(
                row["username"], row["first_name"], row["last_name"]
            )
            messages.append(ContextMessage(speaker=persona_name, text=text))
        return messages or None

    def _build_intro_message(self) -> str:
        return (
            f"Привет! Я {self._bot_name}, бот-двоиник. Я изучаю стиль общения участников"
            " и могу отвечать за них.\n\n"
            "Что я умею:\n"
            "• /imitate @username [текст] — ответить в стиле участника.\n"
            "• /imitate_profiles — показать готовые профили.\n"
            "• /auto_imitate_on или /auto_imitate_off — включить или выключить автоимитацию.\n"
            "• /forgetme — удалить мои данные о вас.\n"
            "Просто общайтесь, а я буду учиться на фоне!"
        )


def run_bot(settings: Settings) -> None:
    bot = BotDouble(settings)
    application = bot.build_application()

    application.add_handler(CommandHandler("imitate", bot.imitate))
    application.add_handler(CommandHandler("imitate_profiles", bot.imitate_profiles))
    application.add_handler(CommandHandler("auto_imitate_on", bot.auto_imitate_on))
    application.add_handler(CommandHandler("auto_imitate_off", bot.auto_imitate_off))
    application.add_handler(CommandHandler("dialogue", bot.dialogue))
    application.add_handler(CommandHandler("profile", bot.profile_command))
    application.add_handler(CommandHandler("me", bot.profile_command))
    application.add_handler(CommandHandler("forgetme", bot.forget_me))
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, bot.on_new_members))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text_message))

    application.run_polling(close_loop=False)
