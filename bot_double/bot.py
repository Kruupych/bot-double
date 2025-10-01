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
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar

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
from .burst_manager import BurstManager
from .command_service import CommandService
from .message_pipeline import MessagePipeline
from .config import Settings
from .db import Database
from .imitation import ChainMessage, ImitationChain, ImitationToolkit
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
    normalize_alias,
)

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


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
        self._message_pipeline = MessagePipeline(self)
        self._burst_manager = BurstManager(
            settings,
            should_break=self._message_pipeline.should_break_burst,
            flush_callback=self._message_pipeline.store_burst,
        )
        self._message_pipeline.attach_burst_manager(self._burst_manager)
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
        self._imitation = ImitationToolkit(
            bot_name=self._bot_name,
            bot_username=self._bot_username,
            chain_cache_limit=500,
            answered_cache_limit=1000,
        )
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

        self._commands = CommandService(
            db=self._db,
            run_db=self._run_db,
            invalidate_alias_cache=self._invalidate_alias_cache,
            get_persona_card=self._get_persona_card,
            get_relationship_summary_text=self._get_relationship_summary_text,
            ensure_internal_user=self._ensure_internal_user,
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
        self._imitation.update_bot_identity(
            bot_name=self._bot_name, bot_username=self._bot_username
        )
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
        # Start a watchdog to flush stale bursts in case timers are cancelled
        self._burst_manager.start()

    async def _post_shutdown(self, application: Application) -> None:
        LOGGER.info("Bot shutting down")
        await self._message_pipeline.flush_all_buffers()
        await self._burst_manager.stop()
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
        user_text = self._imitation.prepare_chain_user_text(
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
        chain = self._imitation.create_chain(
            chat.id,
            user_row,
            message.from_user,
            user_text,
            context_messages=self._imitation.collect_initial_context(message),
        )
        await self._handle_imitation_for_user(message, user_row, chain)

    async def imitate_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return
        # Flush any buffered texts for this chat so counters are up-to-date
        await self._message_pipeline.flush_buffers_for_chat(chat.id)

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

    async def on_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None or message.from_user is None:
            return

        capture_result = await self._message_pipeline.capture_message(message)
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
        user_text = self._imitation.prepare_chain_user_text(
            instruction=starter,
            payload=starter,
            descriptor=None,
            persona_row=user_row,
            persona_name=persona_name,
            message=message,
        )
        if not user_text:
            user_text = "Продолжи разговор."
        chain = self._imitation.create_chain(
            chat.id,
            user_row,
            message.from_user,
            user_text,
            context_messages=self._imitation.collect_initial_context(message),
        )
        await self._handle_imitation_for_user(message, user_row, chain)

    async def on_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None or message.from_user is None:
            return

        capture_result = await self._message_pipeline.capture_message(message)
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
        if sample_text:
            sample_text = self._truncate_for_storage(sample_text)
        full_text = self._truncate_for_storage(stripped)
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
                full_text=full_text,
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
                descriptor, payload = self._imitation.split_imitation_remainder(
                    remainder
                )
            if not descriptor:
                descriptor = self._imitation.descriptor_from_prefix(
                    instruction[:idx]
                )
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
            prefix_lower = instruction[: like_match.start()].lower()
            trigger_hints = ("имитируй", "ответ", "скажи", "напиши", "говори")
            if not any(hint in prefix_lower for hint in trigger_hints):
                return None
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

    def _truncate_for_storage(self, text: str) -> str:
        try:
            limit = int(self._settings.max_store_chars)
        except Exception:
            limit = 0
        if limit and limit > 0 and len(text) > limit:
            return (text[:limit]).rstrip() + "\u2026"
        return text

    async def _choose_persona_artifacts(
        self, chat_id: int, user_id: int, samples: list[str]
    ) -> tuple[Optional[str], Optional[str]]:
        mode = await self._run_db(self._db.get_persona_preference, chat_id, user_id)
        # 0=summary, 1=card, 2=auto
        if mode == 0:
            return None, build_style_summary(samples)
        if mode == 1:
            card = await self._get_persona_card(chat_id, user_id)
            return card, None if card else build_style_summary(samples)
        # auto
        card = await self._get_persona_card(chat_id, user_id)
        return card, None if card else build_style_summary(samples)

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
        if not self._imitation.reserve_answer_slot(message):
            return None
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
        self, message: Message, user_row: sqlite3.Row, chain: ImitationChain
    ) -> None:
        if not self._imitation.reserve_answer_slot(message):
            return
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
        persona_card, style_summary = await self._choose_persona_artifacts(
            chat.id, user_id, samples
        )
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
        starter = self._imitation.format_chain_prompt(chain)
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
        normalized_reply = self._imitation.normalize_chain_text(reply_text)
        if normalized_reply:
            chain.messages.append(
                ChainMessage(
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
            self._imitation.register_chain_reference(
                message.chat_id, bot_reply.message_id, chain
            )
        if message.chat_id is not None and message.from_user is not None:
            self._imitation.remember_chain_for_user(
                message.chat_id, message.from_user.id, chain
            )

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

    @property
    def commands(self) -> CommandService:
        return self._commands

    async def forget_me(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or message.from_user is None:
            return
        reset = await self._run_db(self._db.reset_user_data, message.from_user.id)
        # invalidate alias caches globally, since aliases удалены во всех чатах
        self._alias_cache.clear()
        self._alias_display_cache.clear()
        if reset:
            await message.reply_text(
                "Ваши сообщения, связи и карточка персоны удалены. Я буду заново собирать ваш стиль с новых сообщений."
            )
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
    application.add_handler(CommandHandler("profile", bot.commands.profile_command))
    application.add_handler(CommandHandler("me", bot.commands.profile_command))
    application.add_handler(CommandHandler("forgetme", bot.forget_me))
    application.add_handler(CommandHandler("alias", bot.commands.alias_command))
    application.add_handler(CommandHandler("alias_reset", bot.commands.alias_reset_command))
    application.add_handler(CommandHandler("persona_mode", bot.commands.persona_mode_command))
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, bot.on_new_members))
    application.add_handler(MessageHandler(filters.VOICE, bot.on_voice_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text_message))

    application.run_polling(close_loop=False)
