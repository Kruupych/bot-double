from __future__ import annotations

import asyncio
import difflib
import io
import json
import logging
import random
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple, TypeVar

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

from .assistant_tasks import AssistantTaskEngine, TaskResult
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
    normalize_alias,
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


@dataclass
class _ChainMessage:
    speaker: str
    text: str
    is_persona: bool


@dataclass
class _ImitationChain:
    chat_id: int
    persona_id: int
    persona_username: Optional[str]
    persona_first_name: Optional[str]
    persona_last_name: Optional[str]
    persona_name: str
    messages: List[_ChainMessage]


_CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


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
        self._bot_username: Optional[str] = None
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
        self._alias_cache: Dict[int, Dict[str, int]] = {}
        self._alias_display_cache: Dict[int, Dict[int, List[str]]] = {}
        self._transcriber: Optional[SpeechTranscriber] = None
        self._assistant_tasks: Optional[AssistantTaskEngine] = None
        self._recent_imitation_targets: Dict[Tuple[int, int], int] = {}
        self._chain_by_message: Dict[Tuple[int, int], _ImitationChain] = {}
        self._last_chain_by_user: Dict[Tuple[int, int], _ImitationChain] = {}
        self._chain_cache_limit = 500
        if self._settings.enable_voice_transcription:
            self._transcriber = SpeechTranscriber(
                settings.openai_api_key,
                settings.voice_transcription_model,
                language=settings.voice_transcription_language,
            )
        if self._settings.enable_freeform_intents:
            self._assistant_tasks = AssistantTaskEngine(
                settings.openai_api_key,
                settings.openai_model,
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
        self._bot_username = me.username
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

        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
            return
        persona_name = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        descriptor = f"@{username}"
        instruction_source = starter or message.text or ""
        user_text = self._prepare_chain_user_text(
            instruction=instruction_source,
            payload=starter or None,
            descriptor=descriptor,
            persona_row=user_row,
            persona_name=persona_name,
            message=message,
        )
        if not user_text:
            await message.reply_text(
                "Нужен текст для имитации — добавьте подсказку после команды."
            )
            return
        chain = self._create_chain(
            chat.id,
            user_row,
            message.from_user,
            user_text,
        )
        await self._handle_imitation_for_user(message, user_row, chain)

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

    async def imitate_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return
        lines = [
            "Как попросить меня отвечать за других:",
            "• /imitate @username текст — классическая команда.",
            "• В реплае: ‘имитируй @username …’ или ‘ответь как Тимофей …’.",
            "• Без @username — добавьте прозвища через /alias @user имя, nickname.",
            "• После ответа можно продолжать реплаем: ‘согласись’, ‘добавь деталей’, ‘переведи’.",
            "• Голосовые команды тоже работают: ‘двойник, переведи на английский…’.",
            "• Полезные подсказки: ‘перескажи’, ‘перефразируй’, ‘исправь ошибки’, ‘сделай список задач’.",
            "• Итог: чтобы я сработал, обращайтесь по имени/кличке или отвечайте на мой ответ.",
        ]
        await message.reply_text("\n".join(lines), disable_web_page_preview=True)

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

    async def alias_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        text = message.text or ""
        parts = text.strip().split(None, 2)
        if len(parts) < 2:
            await message.reply_text(
                "Использование: /alias @username прозвище1, прозвище2"
            )
            return
        username_token = parts[1]
        if not username_token.startswith("@"):
            await message.reply_text("Первым аргументом должен быть @username")
            return
        username = username_token.lstrip("@")
        alias_section = ""
        if len(parts) == 3:
            alias_section = parts[2]
        else:
            alias_section = (text.partition(username_token)[2] or "").strip()
        if not alias_section:
            await message.reply_text(
                "Добавьте хотя бы одно прозвище после @username."
            )
            return
        aliases = [alias.strip() for alias in alias_section.split(";")]
        aliases = [chunk for alias in aliases for chunk in alias.split(",")]
        aliases = [alias.strip() for alias in aliases if alias.strip()]
        if not aliases:
            await message.reply_text(
                "Не удалось найти прозвища. Разделяйте их запятыми."
            )
            return
        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
            return
        internal_id = int(user_row["id"])
        added, skipped = await self._run_db(
            self._db.add_aliases,
            chat.id,
            internal_id,
            aliases,
        )
        self._invalidate_alias_cache(chat.id)
        lines: List[str] = []
        if added:
            lines.append(
                "Добавил прозвища: "
                + ", ".join(list(dict.fromkeys(added)))
            )
        if skipped:
            lines.append(
                "Пропущено (уже есть или пустые): "
                + ", ".join(list(dict.fromkeys(skipped)))
            )
        if not lines:
            lines.append("Ничего не добавлено.")
        await message.reply_text("\n".join(lines))

    async def alias_reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        if not context.args:
            await message.reply_text("Использование: /alias_reset @username")
            return
        username_token = context.args[0]
        if not username_token.startswith("@"):
            await message.reply_text("Укажите @username для сброса прозвищ")
            return
        username = username_token.lstrip("@")
        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
            return
        internal_id = int(user_row["id"])
        deleted = await self._run_db(
            self._db.delete_aliases,
            chat.id,
            internal_id,
        )
        self._invalidate_alias_cache(chat.id)
        if deleted:
            await message.reply_text(
                f"Удалено {deleted} прозвищ для @{username}."
            )
        else:
            await message.reply_text(
                f"Для @{username} не было сохранённых прозвищ."
            )

    async def on_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None or message.from_user is None:
            return

        capture_result = await self._capture_message(message)
        processed_text: Optional[str] = None
        from_voice = False
        if capture_result:
            processed_text, from_voice = capture_result

        if processed_text and await self._maybe_handle_intent(
            message, processed_text, from_voice
        ):
            return

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
        persona_name = display_name(
            user_row["username"],
            user_row["first_name"],
            user_row["last_name"],
        )
        user_text = self._prepare_chain_user_text(
            instruction=starter,
            payload=starter,
            descriptor=None,
            persona_row=user_row,
            persona_name=persona_name,
            message=message,
        )
        if not user_text:
            user_text = "Продолжи разговор."
        chain = self._create_chain(
            chat.id,
            user_row,
            message.from_user,
            user_text,
        )
        await self._handle_imitation_for_user(message, user_row, chain)

    async def on_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None or message.from_user is None:
            return

        capture_result = await self._capture_message(message)
        if not capture_result:
            return
        text, from_voice = capture_result
        if await self._maybe_handle_intent(message, text, from_voice):
            return

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
    async def _capture_message(
        self, message: Message
    ) -> Optional[Tuple[str, bool]]:
        if message.from_user is None:
            return None
        user = message.from_user
        if user.is_bot:
            if self._bot_id is None or user.id != self._bot_id:
                return None
        if message.via_bot is not None:
            return None
        if message.forward_origin is not None:
            return None
        chat_id = message.chat_id
        if chat_id is None:
            return None
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
            return None
        text = text.strip()
        if not text:
            return None

        if text_source_voice:
            if text.startswith(('/', '!', '.')):
                await self._flush_buffer_for_key(key)
                self._record_chat_event(message, timestamp)
                return None
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
        return text, text_source_voice

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

    async def _get_alias_maps(
        self, chat_id: int
    ) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        if chat_id in self._alias_cache and chat_id in self._alias_display_cache:
            return self._alias_cache[chat_id], self._alias_display_cache[chat_id]
        rows = await self._run_db(self._db.get_aliases_for_chat, chat_id)
        alias_map: Dict[str, int] = {}
        display_map: Dict[int, List[str]] = {}
        for row in rows:
            normalized = str(row["normalized_alias"])
            user_id = int(row["user_id"])
            alias_map[normalized] = user_id
            display_map.setdefault(user_id, []).append(str(row["alias"]))
        self._alias_cache[chat_id] = alias_map
        self._alias_display_cache[chat_id] = display_map
        return alias_map, display_map

    def _transliterate(self, text: str) -> str:
        result_chars: List[str] = []
        for char in text:
            result_chars.append(_CYR_TO_LAT.get(char, char))
        return "".join(result_chars)

    def _generate_variants(self, values: Iterable[str]) -> Set[str]:
        variants: Set[str] = set()
        for value in values:
            normalized = normalize_alias(value)
            if not normalized:
                continue
            variants.add(normalized)
            translit = self._transliterate(normalized)
            if translit and translit != normalized:
                variants.add(translit)
        return variants

    def _candidate_keys_for_user(
        self, row: sqlite3.Row, aliases: Optional[List[str]]
    ) -> Set[str]:
        values: List[str] = []
        username = row["username"]
        first_name = row["first_name"]
        last_name = row["last_name"]
        if username:
            values.append(username)
            values.append(f"@{username}")
        if first_name:
            values.append(first_name)
        if last_name:
            values.append(last_name)
        if first_name and last_name:
            values.append(f"{first_name} {last_name}")
        display = display_name(username, first_name, last_name)
        if display:
            values.append(display)
        if aliases:
            values.extend(aliases)
        return self._generate_variants(values)

    def _descriptor_variants(self, descriptor: str) -> Set[str]:
        cleaned = descriptor.strip()
        cleaned = re.sub(r"[\"'“”«»]+", "", cleaned)
        candidates: List[str] = []
        parts = re.split(r"[:;,\n\-–—]", cleaned, maxsplit=1)
        base = parts[0].strip()
        if base:
            candidates.append(base)
        tokens = base.split()
        if len(tokens) > 1:
            for length in range(len(tokens), 0, -1):
                segment = " ".join(tokens[:length]).strip()
                if segment and segment not in candidates:
                    candidates.append(segment)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
        variants: Set[str] = set()
        for candidate in candidates:
            normalized = normalize_alias(candidate)
            if not normalized:
                continue
            variants.add(normalized)
            translit = self._transliterate(normalized)
            if translit and translit != normalized:
                variants.add(translit)
        return variants

    def _score_descriptor(
        self, descriptor_variants: Iterable[str], candidate_keys: Iterable[str]
    ) -> float:
        best = 0.0
        candidate_list = list(candidate_keys)
        if not candidate_list:
            return 0.0
        for descriptor in descriptor_variants:
            tokens1 = set(descriptor.split())
            for candidate in candidate_list:
                if descriptor == candidate:
                    return 1.0
                if descriptor and candidate and (
                    descriptor in candidate or candidate in descriptor
                ):
                    best = max(best, 0.92)
                    continue
                tokens2 = set(candidate.split())
                if tokens1 and tokens1 <= tokens2:
                    best = max(best, 0.88)
                    continue
                if tokens1 and tokens2:
                    overlap = tokens1 & tokens2
                    if overlap:
                        ratio = len(overlap) / max(len(tokens1), len(tokens2))
                        best = max(best, 0.65 + 0.25 * ratio)
                        continue
                similarity = difflib.SequenceMatcher(None, descriptor, candidate).ratio()
                if similarity >= 0.75:
                    best = max(best, similarity * 0.85)
        return best

    async def _resolve_user_descriptor(
        self, chat_id: Optional[int], descriptor: str
    ) -> Tuple[Optional[sqlite3.Row], List[Tuple[sqlite3.Row, float]]]:
        if not descriptor:
            return None, []
        descriptor = descriptor.strip()
        if descriptor.startswith("@"):
            username = descriptor.lstrip("@")
            row = await self._run_db(self._db.get_user_by_username, username)
            return row, []
        if chat_id is None:
            return None, []

        alias_map, alias_display = await self._get_alias_maps(chat_id)
        descriptor_variants = self._descriptor_variants(descriptor)

        for variant in list(descriptor_variants):
            if variant in alias_map:
                user_id = alias_map[variant]
                row = await self._fetch_user_row(chat_id, user_id)
                if row:
                    return row, []

        participants = await self._run_db(self._db.get_chat_participants, chat_id)
        rows_by_id: Dict[int, sqlite3.Row] = {
            int(row["id"]): row for row in participants
        }
        # include aliased users who may not have messages yet
        for user_id in alias_display.keys():
            if user_id not in rows_by_id:
                row = await self._run_db(self._db.get_user_by_id, user_id)
                if row:
                    rows_by_id[user_id] = row

        best_candidates: List[Tuple[sqlite3.Row, float]] = []
        best_score = 0.0
        for user_id, row in rows_by_id.items():
            if self._bot_user_id is not None and user_id == self._bot_user_id:
                continue
            aliases = alias_display.get(user_id)
            candidate_keys = self._candidate_keys_for_user(row, aliases)
            score = self._score_descriptor(descriptor_variants, candidate_keys)
            if score < 0.55:
                continue
            if score > best_score + 0.05:
                best_score = score
                best_candidates = [(row, score)]
            elif abs(score - best_score) <= 0.05:
                best_candidates.append((row, score))

        if best_score >= 0.78 and len(best_candidates) == 1:
            return best_candidates[0][0], []

        if best_candidates:
            best_candidates.sort(key=lambda item: item[1], reverse=True)
            return None, best_candidates[:3]

        return None, []

    async def _fetch_user_row(self, chat_id: int, user_id: int) -> Optional[sqlite3.Row]:
        return await self._run_db(self._db.get_user_by_id, user_id)

    def _parse_imitation_request(
        self, instruction: str
    ) -> Optional[Tuple[str, Optional[str]]]:
        lowered = instruction.lower()
        triggers = ["имитируй", "ответь как", "что бы сказал"]
        for trigger in triggers:
            idx = lowered.find(trigger)
            if idx == -1:
                continue
            remainder = instruction[idx + len(trigger) :].strip()
            descriptor: Optional[str] = None
            payload: Optional[str] = None
            if remainder:
                descriptor, payload = self._split_imitation_remainder(remainder)
            if not descriptor:
                descriptor = self._descriptor_from_prefix(instruction[:idx])
            if payload is None and trigger == "что бы сказал" and remainder:
                payload = remainder
            if descriptor:
                return descriptor, payload
        # Patterns like "от лица Тимофей" or "как Тимофей"
        role_match = re.search(r"от лица\s+([^,.;\n]+)", instruction, re.IGNORECASE)
        if role_match:
            descriptor = role_match.group(1).strip()
            payload = (
                (instruction[: role_match.start()].strip())
                + " "
                + instruction[role_match.end() :].strip()
            ).strip()
            payload = payload or None
            return descriptor, payload
        like_match = re.search(r"как\s+([^,.;\n]+)", instruction, re.IGNORECASE)
        if like_match:
            descriptor_raw = like_match.group(1).strip()
            descriptor_tokens = descriptor_raw.split()
            descriptor = descriptor_tokens[0] if descriptor_tokens else descriptor_raw
            payload = (
                (instruction[: like_match.start()].strip())
                + " "
                + instruction[like_match.end() :].strip()
            ).strip()
            return descriptor, (payload or None)
        return None

    def _split_imitation_remainder(
        self, remainder: str
    ) -> Tuple[Optional[str], Optional[str]]:
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
        return remainder.strip(), None

    def _descriptor_from_prefix(self, prefix: str) -> Optional[str]:
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

    async def _maybe_handle_intent(
        self, message: Message, text: str, from_voice: bool
    ) -> bool:
        if not self._settings.enable_freeform_intents or self._assistant_tasks is None:
            return False
        stripped = text.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        direct_address = self._is_direct_address(message, lowered)
        reply_to_bot = (
            message.reply_to_message
            and message.reply_to_message.from_user
            and self._bot_id is not None
            and message.reply_to_message.from_user.id == self._bot_id
        )
        reply_chain: Optional[_ImitationChain] = None
        if (
            reply_to_bot
            and message.reply_to_message
            and message.chat_id is not None
        ):
            reply_chain = self._chain_by_message.get(
                (message.chat_id, message.reply_to_message.message_id)
            )
            if reply_chain and message.from_user is not None:
                self._recent_imitation_targets[
                    (message.chat_id, message.from_user.id)
                ] = reply_chain.persona_id

        cleaned_instruction = self._strip_call_signs(stripped)
        cleaned_lower = cleaned_instruction.lower()
        continuation_chain: Optional[_ImitationChain] = reply_chain
        if (
            continuation_chain is None
            and message.chat_id is not None
            and message.from_user is not None
        ):
            continuation_chain = self._last_chain_by_user.get(
                (message.chat_id, message.from_user.id)
            )
        followup_target_id: Optional[int] = None
        if continuation_chain is not None:
            followup_target_id = continuation_chain.persona_id
        elif message.chat_id is not None and message.from_user is not None:
            followup_target_id = self._recent_imitation_targets.get(
                (message.chat_id, message.from_user.id)
            )

        imitation_request = self._parse_imitation_request(cleaned_instruction)
        if imitation_request:
            descriptor, inline_payload = imitation_request
            resolved_row, suggestions = await self._resolve_user_descriptor(
                message.chat_id, descriptor
            )
            if resolved_row is None:
                if suggestions:
                    lines = [
                        "Не уверена, кого имитировать. Возможные варианты:",
                    ]
                    alias_display = {}
                    if message.chat is not None:
                        _, alias_display = await self._get_alias_maps(message.chat.id)
                    for candidate, score in suggestions:
                        name = display_name(
                            candidate["username"],
                            candidate["first_name"],
                            candidate["last_name"],
                        )
                        username = candidate["username"]
                        handle = f"@{username}" if username else name
                        aliases = alias_display.get(int(candidate["id"]), [])
                        alias_hint = (
                            f" (алиасы: {', '.join(aliases)})" if aliases else ""
                        )
                        lines.append(f"• {handle}{alias_hint}")
                    lines.append(
                        "Уточните имя или создайте прозвища через /alias @username ..."
                    )
                    await message.reply_text("\n".join(lines))
                else:
                    await message.reply_text(
                        "Не могу понять, кого имитировать. Используйте @username или добавьте алиас через /alias."
                    )
                return True

            payload = inline_payload or self._extract_payload(
                message,
                cleaned_instruction,
                keywords=["имитируй", "ответь как", "что бы сказал"],
            )
            if not payload:
                payload = self._extract_reply_text(message)
            if not payload:
                await message.reply_text(
                    "Нужен текст для имитации — добавьте подсказку или ответьте на сообщение."
                )
                return True
            persona_name = display_name(
                resolved_row["username"],
                resolved_row["first_name"],
                resolved_row["last_name"],
            )
            user_text = self._prepare_chain_user_text(
                instruction=cleaned_instruction or stripped,
                payload=payload,
                descriptor=descriptor,
                persona_row=resolved_row,
                persona_name=persona_name,
                message=message,
            )
            if not user_text:
                await message.reply_text(
                    "Нужен текст для имитации — добавьте подсказку или ответьте на сообщение."
                )
                return True
            chat_id = message.chat_id
            if chat_id is None:
                return True
            chain = self._create_chain(
                chat_id,
                resolved_row,
                message.from_user,
                user_text,
            )
            await self._handle_imitation_for_user(
                message, resolved_row, chain
            )
            return True

        if (
            continuation_chain is not None
            and not self._should_skip_chain(cleaned_lower)
            and await self._handle_followup_imitation(
                message,
                cleaned_instruction,
                cleaned_lower,
                stripped,
                continuation_chain,
                reply_to_bot,
                followup_target_id,
            )
        ):
            return True

        if direct_address:
            handled = await self._handle_implicit_imitation(
                message,
                cleaned_instruction,
                stripped,
            )
            if handled:
                return True

        if not direct_address and not reply_to_bot and message.chat.type != "private":
            return False

        payload: Optional[str]

        if any(token in cleaned_lower for token in ("переведи", "translate")):
            language = self._parse_language(cleaned_lower)
            payload = self._extract_payload(
                message, cleaned_instruction, keywords=["переведи", "translate"]
            )
            if not payload:
                await message.reply_text(
                    "Пришлите текст для перевода или ответьте на сообщение с текстом."
                )
                return True
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.translate,
                payload,
                language,
                prefix=f"Перевод на {language}:",
            )
            return True

        if any(token in cleaned_lower for token in ("перескажи", "резюмируй", "кратко")):
            payload = self._extract_payload(
                message,
                cleaned_instruction,
                keywords=["перескаж", "резюмируй", "кратко"],
            )
            if not payload:
                await message.reply_text(
                    "Чтобы пересказать, ответьте на сообщение или укажите текст после двоеточия."
                )
                return True
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.summarize,
                payload,
                prefix="Кратко:",
            )
            return True

        if any(token in cleaned_lower for token in ("перефраз", "формулир")):
            payload = self._extract_payload(
                message,
                cleaned_instruction,
                keywords=["перефраз", "сформулируй", "формулир"],
            )
            if not payload:
                await message.reply_text(
                    "Добавьте текст для перефразирования или ответьте на сообщение."
                )
                return True
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.paraphrase,
                payload,
                "Перепиши текст своими словами, сохранив смысл.",
            )
            return True

        if "исправь ошибки" in cleaned_lower or (
            "провер" in cleaned_lower and "ошиб" in cleaned_lower
        ):
            payload = self._extract_payload(
                message,
                cleaned_instruction,
                keywords=["исправь", "проверь", "ошиб"],
            )
            if not payload:
                await message.reply_text(
                    "Чтобы исправить ошибки, пришлите текст или ответьте на сообщение."
                )
                return True
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.proofread,
                payload,
            )
            return True

        if "список" in cleaned_lower and "задач" in cleaned_lower:
            payload = self._extract_payload(
                message,
                cleaned_instruction,
                keywords=["список", "задач", "сделай"],
            )
            if not payload:
                await message.reply_text(
                    "Чтобы составить список задач, ответьте на сообщение или добавьте текст после команды."
                )
                return True
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.listify,
                payload,
            )
            return True

        tone_instruction: Optional[str] = None
        if "вежлив" in cleaned_lower:
            tone_instruction = "Сделай текст вежливым и уважительным."
        elif "официал" in cleaned_lower:
            tone_instruction = "Сделай текст официальным и деловым."
        elif "дружелюб" in cleaned_lower:
            tone_instruction = "Сделай текст дружелюбным и тёплым."
        elif "токсич" in cleaned_lower:
            tone_instruction = "Убери токсичность и агрессию, сделай нейтральным."
        elif "добавь" in cleaned_lower and (
            "смай" in cleaned_lower or "эмодз" in cleaned_lower
        ):
            tone_instruction = "Добавь немного уместных эмодзи."
        elif "убери" in cleaned_lower and (
            "смай" in cleaned_lower or "эмодз" in cleaned_lower
        ):
            tone_instruction = "Убери эмодзи и сделай текст нейтральным."

        if tone_instruction:
            payload = self._extract_payload(
                message,
                cleaned_instruction,
                keywords=[
                    "сделай",
                    "убери",
                    "добавь",
                    "вежлив",
                    "дружелюб",
                    "токсич",
                    "эмодз",
                    "смай",
                ],
            )
            if not payload:
                await message.reply_text(
                    "Чтобы изменить тон, добавьте текст или ответьте на сообщение."
                )
                return True
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.paraphrase,
                payload,
                tone_instruction,
            )
            return True

        if (
            ("что ответить" in cleaned_lower)
            or ("сформулируй ответ" in cleaned_lower)
            or ("помоги ответить" in cleaned_lower)
        ):
            payload = self._extract_payload(
                message,
                cleaned_instruction,
                keywords=["что ответить", "сформулируй", "ответ"],
            )
            if not payload:
                await message.reply_text(
                    "Чтобы помочь с ответом, добавьте текст сообщения или ответьте на него."
                )
                return True
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.respond_helpfully,
                payload,
                "Ответь кратко и по существу.",
            )
            return True

        if direct_address and "?" in stripped and len(stripped) >= 3:
            await self._execute_assistant_task(
                message,
                self._assistant_tasks.respond_helpfully,
                stripped,
                "Ответь честно и по-человечески.",
            )
            return True

        return False

    async def _handle_implicit_imitation(
        self,
        message: Message,
        cleaned_instruction: str,
        stripped: str,
    ) -> bool:
        chat = message.chat
        if chat is None:
            return False
        descriptor, remainder = self._extract_leading_descriptor(cleaned_instruction)
        if not descriptor:
            return False
        resolved_row, suggestions = await self._resolve_user_descriptor(
            chat.id, descriptor
        )
        if resolved_row is None:
            return False
        payload = remainder or self._extract_payload(
            message, cleaned_instruction
        )
        persona_name = display_name(
            resolved_row["username"],
            resolved_row["first_name"],
            resolved_row["last_name"],
        )
        user_text = self._prepare_chain_user_text(
            instruction=cleaned_instruction or stripped,
            payload=payload,
            descriptor=descriptor,
            persona_row=resolved_row,
            persona_name=persona_name,
            message=message,
        )
        if not user_text:
            return False
        chain = self._create_chain(
            chat.id,
            resolved_row,
            message.from_user,
            user_text,
        )
        await self._handle_imitation_for_user(message, resolved_row, chain)
        return True

    async def _handle_followup_imitation(
        self,
        message: Message,
        cleaned_instruction: str,
        cleaned_lower: str,
        stripped: str,
        base_chain: Optional[_ImitationChain],
        reply_to_bot: bool,
        fallback_persona_id: Optional[int],
    ) -> bool:
        chain_source = base_chain
        persona_row: Optional[sqlite3.Row] = None
        if chain_source is not None:
            persona_row = await self._run_db(
                self._db.get_user_by_id, chain_source.persona_id
            )
            if persona_row is None:
                return False
        elif fallback_persona_id is not None:
            persona_row = await self._run_db(
                self._db.get_user_by_id, fallback_persona_id
            )
            if persona_row is None or message.chat_id is None:
                return False
            persona_name = display_name(
                persona_row["username"],
                persona_row["first_name"],
                persona_row["last_name"],
            )
            chain_source = _ImitationChain(
                chat_id=message.chat_id,
                persona_id=int(persona_row["id"]),
                persona_username=persona_row["username"],
                persona_first_name=persona_row["first_name"],
                persona_last_name=persona_row["last_name"],
                persona_name=persona_name,
                messages=[],
            )
        else:
            return False

        persona_name = chain_source.persona_name
        followup_markers = (
            "соглас",
            "подроб",
            "добав",
            "продолж",
            "опиши",
            "расска",
            "скажи",
            "ответ",
            "согласись",
            "подтверди",
        )

        if not (reply_to_bot or any(marker in cleaned_lower for marker in followup_markers)):
            return False

        payload = self._extract_payload(
            message,
            cleaned_instruction,
            keywords=["имитируй", "ответ", "скажи", "соглас"],
        )
        user_text = self._prepare_chain_user_text(
            instruction=cleaned_instruction,
            payload=payload,
            descriptor=None,
            persona_row=persona_row,
            persona_name=persona_name,
            message=message,
        )
        if not user_text:
            return False

        chain = self._branch_chain(chain_source, message.from_user, user_text)
        await self._handle_imitation_for_user(message, persona_row, chain)
        return True

    def _is_direct_address(self, message: Message, lowered_text: str) -> bool:
        chat = message.chat
        if chat and chat.type == "private":
            return True
        if (
            message.reply_to_message
            and message.reply_to_message.from_user
            and self._bot_id is not None
            and message.reply_to_message.from_user.id == self._bot_id
        ):
            return True
        if self._bot_username and f"@{self._bot_username.lower()}" in lowered_text:
            return True
        name_token = (self._bot_name or "").lower()
        if name_token and name_token in lowered_text:
            return True
        if lowered_text.startswith("двойник") or lowered_text.startswith("бот" + " "):
            return True
        if "двойник" in lowered_text:
            return True
        if "бот-двойник" in lowered_text:
            return True
        return False

    def _strip_call_signs(self, text: str) -> str:
        patterns = [r"^бот[:,\s]*", r"^двойник[:,\s]*"]
        if self._bot_username:
            patterns.append(rf"@{re.escape(self._bot_username)}")
        if self._bot_name:
            patterns.append(re.escape(self._bot_name))
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _extract_reply_text(self, message: Message) -> Optional[str]:
        reply = message.reply_to_message
        if reply is None:
            return None
        if reply.text:
            return reply.text.strip()
        if reply.caption:
            return reply.caption.strip()
        return None

    def _extract_payload(
        self, message: Message, text: str, *, keywords: Optional[List[str]] = None
    ) -> Optional[str]:
        inline = self._text_after_delimiter(text)
        if inline:
            return inline
        quoted = self._text_in_quotes(text)
        if quoted:
            return quoted
        cleaned = text
        if keywords:
            for keyword in keywords:
                pattern = re.compile(keyword, re.IGNORECASE)
                match = pattern.search(cleaned)
                if match:
                    cleaned = cleaned[match.end() :]
        cleaned = self._strip_call_signs(cleaned).strip(" ,\n-—")
        if cleaned:
            return cleaned
        reply_text = self._extract_reply_text(message)
        if reply_text:
            return reply_text
        return None

    def _text_after_delimiter(self, text: str) -> Optional[str]:
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

    def _text_in_quotes(self, text: str) -> Optional[str]:
        match = re.search(r'["«](.+?)["»]', text)
        if match:
            return match.group(1).strip()
        return None

    def _clean_imitation_instruction(
        self,
        instruction: Optional[str],
        descriptor: Optional[str],
        persona_row: sqlite3.Row,
        persona_name: str,
    ) -> Optional[str]:
        if not instruction:
            return None
        text = self._strip_call_signs(instruction).strip()
        text = self._strip_command_prefix(text)
        text = self._remove_descriptor_mentions(text, descriptor, persona_row, persona_name)
        return text or None

    def _clean_imitation_payload(
        self,
        payload: Optional[str],
        persona_row: sqlite3.Row,
        persona_name: str,
    ) -> Optional[str]:
        if not payload:
            return None
        text = self._strip_call_signs(payload).strip()
        text = self._remove_descriptor_mentions(text, None, persona_row, persona_name)
        return text or None

    def _extract_leading_descriptor(
        self, instruction: str
    ) -> Tuple[Optional[str], Optional[str]]:
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

    def _strip_command_prefix(self, text: str) -> str:
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

    def _remove_descriptor_mentions(
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

    def _clone_chain(self, chain: _ImitationChain) -> _ImitationChain:
        return _ImitationChain(
            chat_id=chain.chat_id,
            persona_id=chain.persona_id,
            persona_username=chain.persona_username,
            persona_first_name=chain.persona_first_name,
            persona_last_name=chain.persona_last_name,
            persona_name=chain.persona_name,
            messages=[
                _ChainMessage(
                    speaker=msg.speaker,
                    text=msg.text,
                    is_persona=msg.is_persona,
                )
                for msg in chain.messages
            ],
        )

    def _register_chain_reference(
        self, chat_id: int, message_id: int, chain: _ImitationChain
    ) -> None:
        key = (chat_id, message_id)
        self._chain_by_message[key] = self._clone_chain(chain)
        if len(self._chain_by_message) > self._chain_cache_limit:
            oldest_key = next(iter(self._chain_by_message))
            if oldest_key != key:
                self._chain_by_message.pop(oldest_key, None)

    def _remember_chain_for_user(
        self, chat_id: int, user_id: int, chain: _ImitationChain
    ) -> None:
        key = (chat_id, user_id)
        self._last_chain_by_user[key] = self._clone_chain(chain)
        if len(self._last_chain_by_user) > self._chain_cache_limit:
            oldest_key = next(iter(self._last_chain_by_user))
            if oldest_key != key:
                self._last_chain_by_user.pop(oldest_key, None)

    def _normalize_chain_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _prepare_chain_user_text(
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
            cleaned_instruction = self._clean_imitation_instruction(
                instruction, descriptor, persona_row, persona_name
            )
            if cleaned_instruction:
                candidates.append(cleaned_instruction)
        if payload:
            cleaned_payload = self._clean_imitation_payload(
                payload, persona_row, persona_name
            )
            if cleaned_payload:
                candidates.append(cleaned_payload)
        original = message.text or message.caption or ""
        if original:
            stripped_original = self._strip_call_signs(original)
            stripped_original = self._remove_descriptor_mentions(
                stripped_original, descriptor, persona_row, persona_name
            )
            stripped_original = stripped_original.strip()
            if stripped_original:
                candidates.append(stripped_original)
        unique: List[str] = []
        for candidate in candidates:
            normalized = self._normalize_chain_text(candidate)
            if not normalized:
                continue
            if normalized.startswith("/"):
                continue
            if normalized not in unique:
                unique.append(normalized)
        if unique:
            return "\n".join(unique)
        return None

    def _format_chain_prompt(self, chain: _ImitationChain) -> str:
        if not chain.messages:
            return (
                "Участник задаёт вопрос. Ответь в стиле выбранного пользователя и"
                " поддержи диалог."
            )
        history_lines: List[str] = []
        for entry in chain.messages[:-1]:
            speaker = chain.persona_name if entry.is_persona else entry.speaker
            if entry.is_persona:
                speaker = f"{chain.persona_name} (ты)"
            history_lines.append(f"{speaker}: {entry.text}")
        current = chain.messages[-1]
        current_line = (
            f"{current.speaker} пишет тебе: \"{current.text}\". Ответь как"
            f" {chain.persona_name} от первого лица и продолжи цепочку."
        )
        if history_lines:
            return "Контекст цепочки:\n" + "\n".join(history_lines) + "\n\n" + current_line
        return current_line

    def _create_chain(
        self,
        chat_id: int,
        persona_row: sqlite3.Row,
        requester: Optional[User],
        user_text: str,
    ) -> _ImitationChain:
        persona_name = display_name(
            persona_row["username"],
            persona_row["first_name"],
            persona_row["last_name"],
        )
        chain = _ImitationChain(
            chat_id=chat_id,
            persona_id=int(persona_row["id"]),
            persona_username=persona_row["username"],
            persona_first_name=persona_row["first_name"],
            persona_last_name=persona_row["last_name"],
            persona_name=persona_name,
            messages=[],
        )
        requester_name = "Собеседник"
        if requester is not None:
            requester_name = display_name(
                requester.username, requester.first_name, requester.last_name
            )
        normalized = self._normalize_chain_text(user_text)
        if normalized:
            chain.messages.append(
                _ChainMessage(
                    speaker=requester_name,
                    text=normalized,
                    is_persona=False,
                )
            )
        return chain

    def _branch_chain(
        self,
        base_chain: _ImitationChain,
        requester: Optional[User],
        user_text: str,
    ) -> _ImitationChain:
        chain = self._clone_chain(base_chain)
        requester_name = "Собеседник"
        if requester is not None:
            requester_name = display_name(
                requester.username, requester.first_name, requester.last_name
            )
        normalized = self._normalize_chain_text(user_text)
        if normalized:
            chain.messages.append(
                _ChainMessage(
                    speaker=requester_name,
                    text=normalized,
                    is_persona=False,
                )
            )
        return chain

    def _should_skip_chain(self, cleaned_lower: str) -> bool:
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

    def _parse_language(self, lowered_text: str) -> str:
        language_hints = {
            "англ": "английский",
            "english": "English",
            "рус": "русский",
            "span": "Spanish",
            "испань": "испанский",
            "нем": "немецкий",
            "german": "German",
            "франц": "французский",
            "french": "French",
            "италь": "итальянский",
            "chinese": "Chinese",
            "китай": "китайский",
        }
        for key, value in language_hints.items():
            if key in lowered_text:
                return value
        return "английский"

    async def _run_assistant_task(self, func: Callable[..., TaskResult], *args: object) -> str:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, partial(func, *args))
        return result.text

    async def _execute_assistant_task(
        self,
        message: Message,
        func: Callable[..., TaskResult],
        *args: object,
        prefix: Optional[str] = None,
    ) -> Optional[str]:
        try:
            result = await self._run_assistant_task(func, *args)
        except Exception as exc:
            LOGGER.exception("Assistant task failed", exc_info=exc)
            await message.reply_text(
                "Не удалось выполнить запрос к модели. Попробуйте ещё раз позже."
            )
            return None
        text = result.strip()
        if not text:
            await message.reply_text("Модель не вернула результат.")
            return None
        if prefix:
            await message.reply_text(f"{prefix}\n{text}", disable_web_page_preview=True)
        else:
            await message.reply_text(text, disable_web_page_preview=True)
        return text

    async def _handle_imitation_for_user(
        self, message: Message, user_row: sqlite3.Row, chain: _ImitationChain
    ) -> None:
        chat = message.chat
        if chat is None:
            return
        user_id = int(user_row["id"])
        display = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        message_count = await self._run_db(
            self._db.get_message_count, chat.id, user_id
        )
        if message_count < self._settings.min_messages_for_profile:
            await message.reply_text(
                f"Мне нужно больше сообщений {display}, чтобы имитировать его стиль."
            )
            return
        topic_hint = chain.messages[-1].text if chain.messages else ""
        samples = await self._collect_style_samples(
            chat.id, user_id, topic_hint=topic_hint
        )
        if not samples:
            await message.reply_text(
                f"Сообщений {display} пока недостаточно для генерации ответа."
            )
            return
        persona_card = await self._get_persona_card(chat.id, user_id)
        style_summary = None if persona_card else build_style_summary(samples)
        persona_name = chain.persona_name
        context_messages = None
        peer_profiles = await self._collect_peer_profiles(chat.id, user_id)
        requester_profile = await self._collect_requester_profile(
            chat.id, message.from_user, user_id
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
            user_id,
            addressee_internal_id,
            addressee_name,
        )
        starter = self._format_chain_prompt(chain)
        try:
            reply_text = await self._generate_reply(
                user_row["username"] or "",
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
        except Exception:
            LOGGER.exception("Failed to generate imitation response")
            await message.reply_text(
                "Не удалось сгенерировать ответ. Попробуйте позже или уточните подсказку."
            )
            return
        normalized_reply = self._normalize_chain_text(reply_text)
        if normalized_reply:
            chain.messages.append(
                _ChainMessage(
                    speaker=chain.persona_name,
                    text=normalized_reply,
                    is_persona=True,
                )
            )
        bot_reply = await message.reply_text(reply_text)
        persona_id = int(user_row["id"])
        if message.chat_id is not None and message.from_user is not None:
            self._recent_imitation_targets[
                (message.chat_id, message.from_user.id)
            ] = persona_id
        if message.chat_id is not None and bot_reply is not None:
            self._register_chain_reference(message.chat_id, bot_reply.message_id, chain)
        if message.chat_id is not None and message.from_user is not None:
            self._remember_chain_for_user(message.chat_id, message.from_user.id, chain)

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

    def _invalidate_alias_cache(self, chat_id: Optional[int]) -> None:
        if chat_id is None:
            return
        self._alias_cache.pop(chat_id, None)
        self._alias_display_cache.pop(chat_id, None)

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
            file = await voice.get_file()
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
    application.add_handler(CommandHandler("imitate_help", bot.imitate_help))
    application.add_handler(CommandHandler("auto_imitate_on", bot.auto_imitate_on))
    application.add_handler(CommandHandler("auto_imitate_off", bot.auto_imitate_off))
    application.add_handler(CommandHandler("dialogue", bot.dialogue))
    application.add_handler(CommandHandler("profile", bot.profile_command))
    application.add_handler(CommandHandler("me", bot.profile_command))
    application.add_handler(CommandHandler("forgetme", bot.forget_me))
    application.add_handler(CommandHandler("alias", bot.alias_command))
    application.add_handler(CommandHandler("alias_reset", bot.alias_reset_command))
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, bot.on_new_members))
    application.add_handler(MessageHandler(filters.VOICE, bot.on_voice_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text_message))

    application.run_polling(close_loop=False)
