from __future__ import annotations

import asyncio
import logging
import random
import re
from functools import partial
from typing import Any, Callable, List, Optional, Set, TypeVar

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
from .style_analysis import build_style_summary
from .utils import (
    display_name,
    guess_gender,
    should_store_context_snippet,
    should_store_message,
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
        self._bot_id: Optional[int] = None
        self._bot_name: str = "Бот-Двойник"
        self._bot_user_id: Optional[int] = None

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

    async def _post_shutdown(self, application: Application) -> None:
        LOGGER.info("Bot shutting down")
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

        style_summary = build_style_summary(samples)

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

        style_summary = build_style_summary(samples)

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

        style_summary_a = build_style_summary(samples_a)
        style_summary_b = build_style_summary(samples_b)

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
                relationship_hint_a,
                username_b,
                persona_name_b,
                samples_b,
                style_summary_b,
                relationship_hint_b,
                topic or ""
            )
        except Exception as exc:  # pragma: no cover - network errors etc.
            LOGGER.exception("Failed to generate dialogue", exc_info=exc)
            await message.reply_text("Не удалось построить диалог, попробуйте позже")
            return

        await message.reply_text(dialogue_text)

    # --- internal helpers -----------------------------------------------------------
    async def _capture_message(self, message: Message) -> None:
        if message.from_user is None:
            return
        user = message.from_user
        text = message.text or ""
        if not text:
            return
        if user.is_bot:
            if self._bot_id is None or user.id != self._bot_id:
                return
        if message.via_bot is not None:
            return
        if message.forward_origin is not None:
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
        if should_store_message(
            message,
            min_tokens=self._settings.min_tokens_to_store,
            allowed_bot_id=self._bot_id,
        ):
            await self._run_db(
                self._db.store_message,
                message.chat_id,
                user_id,
                text,
                timestamp,
                context_only=False,
            )

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

        await self._update_pair_interactions(message, user_id, text)

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
        )
        return build_relationship_hint(addressee_name, relationship_stats)

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
            relationship_hint,
        )

    async def _generate_dialogue(
        self,
        username_a: str,
        persona_name_a: str,
        samples_a: List[str],
        style_summary_a: Optional[str],
        relationship_hint_a: Optional[str],
        username_b: str,
        persona_name_b: str,
        samples_b: List[str],
        style_summary_b: Optional[str],
        relationship_hint_b: Optional[str],
        topic: str,
    ) -> str:
        loop = asyncio.get_running_loop()
        participant_a = DialogueParticipant(
            username=username_a,
            name=persona_name_a,
            samples=[StyleSample(text=sample) for sample in samples_a],
            style_summary=style_summary_a,
            relationship_hint=relationship_hint_a,
        )
        participant_b = DialogueParticipant(
            username=username_b,
            name=persona_name_b,
            samples=[StyleSample(text=sample) for sample in samples_b],
            style_summary=style_summary_b,
            relationship_hint=relationship_hint_b,
        )
        return await loop.run_in_executor(
            None,
            self._style.generate_dialogue,
            participant_a,
            participant_b,
            topic,
        )

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
    application.add_handler(CommandHandler("forgetme", bot.forget_me))
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, bot.on_new_members))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text_message))

    application.run_polling(close_loop=False)
