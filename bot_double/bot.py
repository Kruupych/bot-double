from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import re
import sqlite3
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from telegram import Message, Update, User
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .burst_manager import BurstManager
from .command_service import CommandService
from .config import Settings
from .db import Database
from .imitation import ImitationChain, ImitationToolkit
from .imitation_service import ImitationService
from .message_pipeline import MessagePipeline
from .persona_worker import PersonaAnalysisWorker
from .relationship_worker import RelationshipAnalysisWorker
from .style_engine import (
    ContextMessage,
    ParticipantProfile,
    RequesterProfile,
    StyleEngine,
)
from .relationship_analysis import (
    RelationshipStats,
    build_relationship_hint,
    evaluate_interaction,
)
from .social_analysis import SocialAnalyzer
from .persona_analysis import PersonaAnalyzer
from .style_analysis import build_style_summary
from .transcription import SpeechTranscriber
from .user_resolver import UserResolverService
from .utils import (
    display_name,
    guess_gender,
)

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


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
        self._relationship_worker = RelationshipAnalysisWorker(
            settings=settings,
            db=self._db,
            run_db=self._run_db,
            social_analyzer=self._social,
        )
        self._persona_worker: Optional[PersonaAnalysisWorker] = (
            PersonaAnalysisWorker(
                settings=settings,
                db=self._db,
                run_db=self._run_db,
                persona_analyzer=self._persona_analyzer,
            )
            if self._persona_analyzer
            else None
        )
        self._user_resolver = UserResolverService(
            db=self._db,
            run_db=self._run_db,
            get_bot_user_id=lambda: self._bot_user_id,
        )
        self._transcriber: Optional[SpeechTranscriber] = None
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

        self._commands = CommandService(
            settings=self._settings,
            db=self._db,
            run_db=self._run_db,
            invalidate_alias_cache=self._invalidate_alias_cache,
            get_persona_card=self._get_persona_card,
            get_style_summary=self._get_style_summary,
            get_relationship_summary_text=self._get_relationship_summary_text,
            ensure_internal_user=self._ensure_internal_user,
            flush_buffers_for_chat=self._message_pipeline.flush_buffers_for_chat,
        )
        self._imitation_service = ImitationService(
            settings=self._settings,
            db=self._db,
            run_db=self._run_db,
            imitation=self._imitation,
            style_engine=self._style,
            collect_style_samples=self._collect_style_samples,
            choose_persona_artifacts=self._choose_persona_artifacts,
            collect_peer_profiles=self._collect_peer_profiles,
            collect_requester_profile=self._collect_requester_profile,
            relationship_hint_for_addressee=self._relationship_hint_for_addressee,
            ensure_internal_user=self._ensure_internal_user,
            get_persona_card=self._get_persona_card,
            get_dialog_context=self._get_dialog_context,
            resolve_user_descriptor=self._resolve_user_descriptor,
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
        # Start analysis workers
        if not self._relationship_worker.is_running:
            self._relationship_worker.start()
        if self._persona_worker is not None and not self._persona_worker.is_running:
            self._persona_worker.start()
        # Start a watchdog to flush stale bursts in case timers are cancelled
        self._burst_manager.start()

    async def _post_shutdown(self, application: Application) -> None:
        LOGGER.info("Bot shutting down")
        await self._message_pipeline.flush_all_buffers()
        await self._burst_manager.stop()
        await self._relationship_worker.stop()
        if self._persona_worker is not None:
            await self._persona_worker.stop()
        self._db.close()

    # --- handlers ------------------------------------------------------------------
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
        if await self._imitation_service.maybe_auto_imitate(message):
            return

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

        mention_usernames = self._imitation_service.extract_mentions(message)
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

    async def _get_alias_maps(
        self, chat_id: int
    ) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        return await self._user_resolver.get_alias_maps(chat_id)

    async def _resolve_user_descriptor(
        self, chat_id: Optional[int], descriptor: str
    ) -> Tuple[Optional[sqlite3.Row], List[Tuple[sqlite3.Row, float]]]:
        return await self._user_resolver.resolve_user_descriptor(chat_id, descriptor)

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
        if lowered_text.startswith("двойник"):
            return True
        if lowered_text.startswith("бот"):
            if len(lowered_text) == 3:
                return True
            next_char = lowered_text[3]
            if next_char in {" ", ",", ":", ";", "-", "—", "!", "?", "."}:
                return True
        if "бот-двойник" in lowered_text:
            return True
        if "двойник" in lowered_text:
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
        # 0=summary, 1=card, 2=auto, 3=combined
        if mode == 0:
            return None, build_style_summary(samples)
        if mode == 1:
            card = await self._get_persona_card(chat_id, user_id)
            return card, None if card else build_style_summary(samples)
        if mode == 3:
            card = await self._get_persona_card(chat_id, user_id)
            summary = build_style_summary(samples)
            return card, summary
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

    async def _maybe_handle_intent(
        self, message: Message, text: str, from_voice: bool
    ) -> bool:
        if not self._settings.enable_freeform_intents:
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

        reply_chain: Optional[ImitationChain] = None
        if (
            reply_to_bot
            and message.reply_to_message
            and message.chat_id is not None
        ):
            reply_chain = self._imitation_service.get_chain_for_message(
                message.chat_id,
                message.reply_to_message.message_id,
            )
            if reply_chain and message.from_user is not None:
                self._imitation_service.remember_target(
                    message.chat_id, message.from_user.id, reply_chain.persona_id
                )

        cleaned_instruction = self._imitation.strip_call_signs(stripped)
        cleaned_lower = cleaned_instruction.lower()

        continuation_chain = reply_chain
        followup_target_id: Optional[int] = None
        if continuation_chain is not None:
            followup_target_id = continuation_chain.persona_id
        elif message.chat_id is not None and message.from_user is not None:
            followup_target_id = self._imitation_service.get_recent_target(
                message.chat_id, message.from_user.id
            )

        imitation_request = self._parse_imitation_request(cleaned_instruction)
        if imitation_request:
            descriptor, inline_payload = imitation_request
            resolved_row, suggestions = await self._resolve_user_descriptor(
                message.chat_id, descriptor
            )
            if resolved_row is None:
                if suggestions:
                    lines = ["Не уверена, кого имитировать. Возможные варианты:"]
                    alias_display = {}
                    if message.chat is not None:
                        _, alias_display = await self._get_alias_maps(message.chat.id)
                    for candidate, _score in suggestions:
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

            payload = inline_payload or self._imitation.extract_payload(
                message,
                cleaned_instruction,
                keywords=["имитируй", "ответь как", "что бы сказал"],
            )
            if not payload:
                payload = self._imitation.extract_reply_text(message)
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
            user_text = self._imitation.prepare_chain_user_text(
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

            chain = self._imitation.create_chain(
                chat_id,
                resolved_row,
                message.from_user,
                user_text,
                context_messages=self._imitation.collect_initial_context(message),
            )
            await self._imitation_service.handle_chain(message, resolved_row, chain)
            return True

        if (
            (continuation_chain is not None or followup_target_id is not None)
            and await self._imitation_service.maybe_handle_followup(
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

        if (
            direct_address
            and not self._imitation.should_skip_chain(cleaned_lower)
            and await self._imitation_service.maybe_handle_direct_imitation(
                message, cleaned_instruction, stripped
            )
        ):
            return True

        if (
            not direct_address
            and not reply_to_bot
            and message.chat is not None
            and message.chat.type != "private"
        ):
            return False

        return False

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
        persona_card = "\n".join(lines).strip()
        return persona_card or None

    async def _get_style_summary(self, chat_id: int, user_id: int) -> Optional[str]:
        samples = await self._collect_style_samples(chat_id, user_id)
        if not samples:
            return None
        summary = build_style_summary(samples).strip()
        return summary or None

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
        if self._persona_analyzer is None or self._persona_worker is None:
            return
        if self._bot_user_id is not None and user_id == self._bot_user_id:
            return
        await self._run_db(self._db.increment_persona_pending, chat_id, user_id)
        await self._maybe_queue_persona_analysis(chat_id, user_id)

    async def _maybe_queue_relationship_analysis(
        self, chat_id: int, speaker_id: int, target_id: int
    ) -> None:
        await self._relationship_worker.maybe_queue((chat_id, speaker_id, target_id))

    async def _maybe_queue_persona_analysis(self, chat_id: int, user_id: int) -> None:
        if self._persona_worker is None:
            return
        await self._persona_worker.maybe_queue((chat_id, user_id))

    def _invalidate_alias_cache(self, chat_id: Optional[int]) -> None:
        self._user_resolver.invalidate_cache(chat_id)

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
        if is_same_person:
            _log.info(
                "Self-imitation detected: requester_id=%d, target_id=%d, user=%s",
                requester_internal_id, target_internal_id, name,
            )
        if not formatted and not is_same_person:
            return None
        # Fetch requester aliases (only if not same person)
        requester_aliases: Optional[List[str]] = None
        if not is_same_person:
            requester_aliases = await self._run_db(
                self._db.get_user_aliases, chat_id, requester_internal_id, 5
            ) or None
        return RequesterProfile(
            name=name,
            samples=formatted[:sample_limit],
            is_same_person=is_same_person,
            aliases=requester_aliases,
        )

    async def _run_db(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        bound = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound)

    @property
    def commands(self) -> CommandService:
        return self._commands

    @property
    def imitation(self) -> ImitationService:
        return self._imitation_service

    async def forget_me(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or message.from_user is None:
            return
        reset = await self._run_db(self._db.reset_user_data, message.from_user.id)
        # invalidate alias caches globally, since aliases удалены во всех чатах
        self._user_resolver.clear_all_caches()
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
            f"Привет! Я {self._bot_name}. Я изучаю стиль общения участников"
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

    application.add_handler(CommandHandler("imitate", bot.imitation.imitate_command))
    application.add_handler(CommandHandler("imitate_profiles", bot.commands.imitate_profiles))
    application.add_handler(CommandHandler("imitate_help", bot.commands.imitate_help))
    application.add_handler(CommandHandler("auto_imitate_on", bot.commands.auto_imitate_on))
    application.add_handler(CommandHandler("auto_imitate_off", bot.commands.auto_imitate_off))
    application.add_handler(CommandHandler("dialogue", bot.imitation.dialogue_command))
    application.add_handler(CommandHandler("roast", bot.imitation.roast_command))
    application.add_handler(CommandHandler("horoscope", bot.imitation.horoscope_command))
    application.add_handler(CommandHandler("compatibility", bot.imitation.compatibility_command))
    application.add_handler(CommandHandler("story", bot.imitation.story_command))
    application.add_handler(CommandHandler("long_story", bot.imitation.long_story_command))
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
