from __future__ import annotations

import asyncio
import random
import re
import sqlite3
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar

from telegram import Message, MessageEntity, Update, User
from telegram.ext import ContextTypes

from .config import Settings
from .db import Database
from .imitation import ChainMessage, ImitationChain, ImitationToolkit
from .style_analysis import build_style_summary
from .style_engine import (
    ContextMessage,
    DialogueParticipant,
    ParticipantProfile,
    RequesterProfile,
    StoryCharacter,
    StyleEngine,
    StyleSample,
)
from .utils import display_name, guess_gender, send_long_text

T = TypeVar("T")
RunDB = Callable[..., Awaitable[T]]
CollectStyleSamples = Callable[..., Awaitable[List[str]]]
ChoosePersonaArtifacts = Callable[[int, int, List[str]], Awaitable[Tuple[Optional[str], Optional[str]]]]
CollectPeerProfiles = Callable[[int, int], Awaitable[Optional[List[ParticipantProfile]]]]
CollectRequesterProfile = Callable[[int, Optional[User], int], Awaitable[Optional[RequesterProfile]]]
RelationshipHintForAddressee = Callable[[int, int, Optional[int], str], Awaitable[Optional[str]]]
EnsureInternalUser = Callable[[Optional[User]], Awaitable[Optional[int]]]
GetPersonaCard = Callable[[int, int], Awaitable[Optional[str]]]
GetDialogContext = Callable[[int], Awaitable[Optional[List[ContextMessage]]]]
ResolveUserDescriptor = Callable[
    [Optional[int], str],
    Awaitable[Tuple[Optional[sqlite3.Row], List[Tuple[sqlite3.Row, float]]]],
]


class ImitationService:
    """Handles imitation-related commands, dialogue generation, and auto replies."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        run_db: RunDB,
        imitation: ImitationToolkit,
        style_engine: StyleEngine,
        collect_style_samples: CollectStyleSamples,
        choose_persona_artifacts: ChoosePersonaArtifacts,
        collect_peer_profiles: CollectPeerProfiles,
        collect_requester_profile: CollectRequesterProfile,
        relationship_hint_for_addressee: RelationshipHintForAddressee,
        ensure_internal_user: EnsureInternalUser,
        get_persona_card: GetPersonaCard,
        get_dialog_context: GetDialogContext,
        resolve_user_descriptor: ResolveUserDescriptor,
    ) -> None:
        self._settings = settings
        self._db = db
        self._run_db = run_db
        self._imitation = imitation
        self._style = style_engine
        self._collect_style_samples = collect_style_samples
        self._choose_persona_artifacts = choose_persona_artifacts
        self._collect_peer_profiles = collect_peer_profiles
        self._collect_requester_profile = collect_requester_profile
        self._relationship_hint_for_addressee = relationship_hint_for_addressee
        self._ensure_internal_user = ensure_internal_user
        self._get_persona_card = get_persona_card
        self._recent_targets: Dict[Tuple[int, int], int] = {}
        self._get_dialog_context = get_dialog_context
        self._resolve_user_descriptor = resolve_user_descriptor

    async def imitate_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    async def dialogue_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
            await message.reply_text("Первые два аргумента должны быть @username участников")
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

        # Fetch aliases for both participants
        aliases_a = await self._run_db(self._db.get_user_aliases, chat.id, internal_a, 5) or None
        aliases_b = await self._run_db(self._db.get_user_aliases, chat.id, internal_b, 5) or None

        try:
            dialogue_text = await self._generate_dialogue(
                username_a,
                persona_name_a,
                samples_a,
                style_summary_a,
                persona_card_a,
                relationship_hint_a,
                aliases_a,
                username_b,
                persona_name_b,
                samples_b,
                style_summary_b,
                persona_card_b,
                relationship_hint_b,
                aliases_b,
                topic or "",
            )
        except Exception:
            await message.reply_text("Не удалось построить диалог, попробуйте позже")
            return

        await message.reply_text(dialogue_text)

    async def roast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate a playful roast of a user based on their communication style."""
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return

        if not context.args:
            await message.reply_text("Использование: /roast @username")
            return

        username_arg = context.args[0]
        if not username_arg.startswith("@"):
            await message.reply_text("Аргументом должен быть @username")
            return

        username = username_arg.lstrip("@")
        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
            return

        user_id = int(user_row["id"])
        persona_name = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )

        message_count = await self._run_db(
            self._db.get_message_count, chat.id, user_id
        )
        if message_count < self._settings.min_messages_for_profile:
            await message.reply_text(
                f"Мне нужно больше сообщений от {persona_name}, "
                f"чтобы составить поджарку (минимум {self._settings.min_messages_for_profile})."
            )
            return

        samples = await self._collect_style_samples(chat.id, user_id, topic_hint="")
        if not samples:
            await message.reply_text(
                f"Сообщений {persona_name} пока недостаточно для генерации поджарки."
            )
            return

        persona_card, style_summary = await self._choose_persona_artifacts(
            chat.id, user_id, samples
        )

        # Fetch aliases for persona
        aliases = await self._run_db(self._db.get_user_aliases, chat.id, user_id, 5) or None

        try:
            roast_text = await self._generate_roast(
                user_row["username"] or username,
                persona_name,
                samples,
                style_summary,
                persona_card,
                aliases,
            )
        except Exception:
            await message.reply_text(
                "Не удалось сгенерировать поджарку. Попробуйте позже."
            )
            return

        header = f"🔥 Поджарка для {persona_name}:\n\n"
        await message.reply_text(header + roast_text)

    async def _generate_roast(
        self,
        username: str,
        persona_name: str,
        samples: List[str],
        style_summary: Optional[str],
        persona_card: Optional[str],
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Call StyleEngine to generate a roast."""
        from .style_engine import StyleSample

        style_samples = [StyleSample(text=s) for s in samples]
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._style.generate_roast(
                username,
                persona_name,
                style_samples,
                style_summary,
                persona_card,
                aliases,
            ),
        )

    async def horoscope_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate a personalized horoscope based on user's communication patterns."""
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return

        if not context.args:
            await message.reply_text("Использование: /horoscope @username")
            return

        username_arg = context.args[0]
        if not username_arg.startswith("@"):
            await message.reply_text("Аргументом должен быть @username")
            return

        username = username_arg.lstrip("@")

        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
            return

        user_id = int(user_row["id"])
        persona_name = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )

        message_count = await self._run_db(
            self._db.get_message_count, chat.id, user_id
        )
        if message_count < self._settings.min_messages_for_profile:
            await message.reply_text(
                f"Мне нужно больше сообщений от {persona_name}, "
                f"чтобы составить гороскоп (минимум {self._settings.min_messages_for_profile})."
            )
            return

        samples = await self._collect_style_samples(chat.id, user_id, topic_hint="")
        if not samples:
            await message.reply_text(
                f"Сообщений {persona_name} пока недостаточно для генерации гороскопа."
            )
            return

        persona_card, style_summary = await self._choose_persona_artifacts(
            chat.id, user_id, samples
        )

        # Fetch aliases for persona
        aliases = await self._run_db(self._db.get_user_aliases, chat.id, user_id, 5) or None

        try:
            horoscope_text = await self._generate_horoscope(
                user_row["username"] or username,
                persona_name,
                samples,
                style_summary,
                persona_card,
                aliases,
            )
        except Exception:
            await message.reply_text(
                "Не удалось сгенерировать гороскоп. Попробуйте позже."
            )
            return

        header = f"🔮 Гороскоп для {persona_name}:\n\n"
        await message.reply_text(header + horoscope_text)

    async def _generate_horoscope(
        self,
        username: str,
        persona_name: str,
        samples: List[str],
        style_summary: Optional[str],
        persona_card: Optional[str],
        aliases: Optional[List[str]] = None,
    ) -> str:
        """Call StyleEngine to generate a horoscope."""
        from .style_engine import StyleSample

        style_samples = [StyleSample(text=s) for s in samples]
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._style.generate_horoscope(
                username,
                persona_name,
                style_samples,
                style_summary,
                persona_card,
                aliases,
            ),
        )

    async def compatibility_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate a fun compatibility test between two users based on communication styles."""
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return

        if len(context.args) < 2:
            await message.reply_text("Использование: /compatibility @user1 @user2")
            return

        username_a_arg, username_b_arg = context.args[0], context.args[1]
        if not username_a_arg.startswith("@") or not username_b_arg.startswith("@"):
            await message.reply_text("Оба аргумента должны быть @username")
            return

        username_a = username_a_arg.lstrip("@")
        username_b = username_b_arg.lstrip("@")

        if username_a.lower() == username_b.lower():
            await message.reply_text("Нужны два разных пользователя для теста совместимости 😄")
            return

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

        persona_name_a = display_name(
            row_a["username"], row_a["first_name"], row_a["last_name"]
        )
        persona_name_b = display_name(
            row_b["username"], row_b["first_name"], row_b["last_name"]
        )

        # Check minimum messages for both users
        count_a = await self._run_db(self._db.get_message_count, chat.id, internal_a)
        count_b = await self._run_db(self._db.get_message_count, chat.id, internal_b)
        min_required = self._settings.min_messages_for_profile

        if count_a < min_required:
            await message.reply_text(
                f"Мне нужно больше сообщений от {persona_name_a} "
                f"(минимум {min_required})."
            )
            return
        if count_b < min_required:
            await message.reply_text(
                f"Мне нужно больше сообщений от {persona_name_b} "
                f"(минимум {min_required})."
            )
            return

        # Collect samples for both users
        samples_a = await self._collect_style_samples(chat.id, internal_a, topic_hint="")
        samples_b = await self._collect_style_samples(chat.id, internal_b, topic_hint="")

        if not samples_a:
            await message.reply_text(f"Недостаточно сообщений от {persona_name_a}.")
            return
        if not samples_b:
            await message.reply_text(f"Недостаточно сообщений от {persona_name_b}.")
            return

        # Get style summaries
        _, style_summary_a = await self._choose_persona_artifacts(
            chat.id, internal_a, samples_a
        )
        _, style_summary_b = await self._choose_persona_artifacts(
            chat.id, internal_b, samples_b
        )

        # Fetch aliases for both users
        aliases_a = await self._run_db(self._db.get_user_aliases, chat.id, internal_a, 5) or None
        aliases_b = await self._run_db(self._db.get_user_aliases, chat.id, internal_b, 5) or None

        try:
            result = await self._generate_compatibility(
                persona_name_a,
                row_a["username"] or username_a,
                samples_a,
                style_summary_a,
                aliases_a,
                persona_name_b,
                row_b["username"] or username_b,
                samples_b,
                style_summary_b,
                aliases_b,
            )
        except Exception:
            await message.reply_text(
                "Не удалось сгенерировать тест совместимости. Попробуйте позже."
            )
            return

        header = f"💕 Тест совместимости: {persona_name_a} & {persona_name_b}\n\n"
        await message.reply_text(header + result)

    async def _generate_compatibility(
        self,
        name_a: str,
        username_a: str,
        samples_a: List[str],
        style_summary_a: Optional[str],
        aliases_a: Optional[List[str]],
        name_b: str,
        username_b: str,
        samples_b: List[str],
        style_summary_b: Optional[str],
        aliases_b: Optional[List[str]],
    ) -> str:
        """Call StyleEngine to generate compatibility analysis."""
        from .style_engine import StyleSample

        style_samples_a = [StyleSample(text=s) for s in samples_a]
        style_samples_b = [StyleSample(text=s) for s in samples_b]

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._style.generate_compatibility(
                name_a,
                username_a,
                style_samples_a,
                style_summary_a,
                aliases_a,
                name_b,
                username_b,
                style_samples_b,
                style_summary_b,
                aliases_b,
            ),
        )

    async def story_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate a short story with chat participants as characters."""
        await self._handle_story_command(update, context, long_version=False)

    async def long_story_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate a longer story with chat participants as characters."""
        await self._handle_story_command(update, context, long_version=True)

    async def _handle_story_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        long_version: bool,
    ) -> None:
        """Common handler for story and long_story commands."""
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return

        # Parse arguments: /story @user1 @user2 ... [тема]
        if not context.args or len(context.args) < 2:
            cmd = "/long_story" if long_version else "/story"
            await message.reply_text(
                f"Использование: {cmd} @user1 @user2 [@user3...] [тема]\n"
                "Минимум 2 участника."
            )
            return

        # Separate usernames from topic
        usernames: List[str] = []
        topic_parts: List[str] = []
        for arg in context.args:
            if arg.startswith("@"):
                usernames.append(arg.lstrip("@"))
            else:
                topic_parts.append(arg)

        if len(usernames) < 2:
            await message.reply_text("Нужно минимум 2 участника (@user1 @user2).")
            return

        if len(usernames) > 5:
            await message.reply_text("Максимум 5 участников для истории.")
            return

        topic = " ".join(topic_parts).strip() if topic_parts else None

        # Resolve all users
        user_rows: List[sqlite3.Row] = []
        for username in usernames:
            row = await self._run_db(self._db.get_user_by_username, username)
            if row is None:
                await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
                return
            user_rows.append(row)

        # Check message counts and collect data
        min_required = self._settings.min_messages_for_profile
        characters: List[StoryCharacter] = []

        for row in user_rows:
            internal_id = int(row["id"])
            persona_name = display_name(
                row["username"], row["first_name"], row["last_name"]
            )

            count = await self._run_db(self._db.get_message_count, chat.id, internal_id)
            if count < min_required:
                await message.reply_text(
                    f"Мне нужно больше сообщений от {persona_name} "
                    f"(минимум {min_required})."
                )
                return

            samples = await self._collect_style_samples(chat.id, internal_id, topic_hint="")
            if not samples:
                await message.reply_text(f"Недостаточно сообщений от {persona_name}.")
                return

            _, style_summary = await self._choose_persona_artifacts(
                chat.id, internal_id, samples
            )

            # Fetch aliases for character
            aliases = await self._run_db(self._db.get_user_aliases, chat.id, internal_id, 5) or None

            characters.append(StoryCharacter(
                name=persona_name,
                username=row["username"] or f"user{internal_id}",
                samples=[StyleSample(text=s) for s in samples],
                style_summary=style_summary,
                aliases=aliases,
            ))

        # Generate the story
        story_type = "развёрнутую историю" if long_version else "историю"
        await message.reply_text(f"✍️ Пишу {story_type}...")

        try:
            story_text = await self._generate_story(
                characters,
                topic,
                long_version=long_version,
            )
        except Exception:
            await message.reply_text(
                "Не удалось сгенерировать историю. Попробуйте позже."
            )
            return

        # Add header
        char_names = " & ".join(c.name for c in characters)
        header = f"📖 {'Рассказ' if long_version else 'История'}: {char_names}\n\n"

        # Send with smart splitting / document fallback
        # Long stories: 8000 chars threshold, short stories: 6000
        doc_threshold = 8000 if long_version else 6000
        await send_long_text(
            message,
            header + story_text,
            document_threshold=doc_threshold,
            document_filename="story.txt",
            document_caption=f"📖 {char_names}",
        )

    async def _generate_story(
        self,
        characters: List[StoryCharacter],
        topic: Optional[str],
        *,
        long_version: bool,
    ) -> str:
        """Call StyleEngine to generate a story."""
        # Get story-specific reasoning effort from settings
        story_reasoning = self._settings.openai_story_reasoning_effort

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._style.generate_story(
                characters,
                topic,
                long_version=long_version,
                reasoning_effort=story_reasoning,
            ),
        )

    async def maybe_auto_imitate(self, message: Message) -> bool:
        chat = message.chat
        if chat is None or message.from_user is None:
            return False
        is_enabled = await self._run_db(self._db.is_auto_imitate_enabled, chat.id)
        if not is_enabled:
            return False

        mention_usernames = self.extract_mentions(message)
        if not mention_usernames:
            return False

        username = self._pick_candidate_username(
            mention_usernames, message.from_user.username
        )
        if not username:
            return False

        user_row = await self._run_db(self._db.get_user_by_username, username)
        if user_row is None:
            return False

        message_count = await self._run_db(
            self._db.get_message_count, chat.id, int(user_row["id"])
        )
        if message_count < self._settings.min_messages_for_profile:
            return False

        if random.random() > self._settings.auto_imitate_probability:
            return False

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
        return True

    def extract_mentions(self, message: Message) -> List[str]:
        entities = message.parse_entities([MessageEntity.MENTION])
        usernames: List[str] = []
        for _, value in entities.items():
            username = value.lstrip("@")
            if username:
                usernames.append(username)
        return usernames

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
        context_messages: Optional[List[ContextMessage]] = None
        if message.chat_id is not None:
            has_persona_history = any(entry.is_persona for entry in chain.messages)
            if not has_persona_history:
                context_messages = await self._get_dialog_context(message.chat_id)
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
        # Fetch random aliases for persona (up to 5)
        persona_aliases = await self._run_db(
            self._db.get_user_aliases, chat.id, user_id, 5
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
                persona_aliases or None,
            )
        except Exception:
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
        if message.chat_id is not None and message.from_user is not None:
            self._recent_targets[(message.chat_id, message.from_user.id)] = user_id
        if message.chat_id is not None and bot_reply is not None:
            self._imitation.register_chain_reference(
                message.chat_id, bot_reply.message_id, chain
            )
        if message.chat_id is not None and message.from_user is not None:
            self._imitation.remember_chain_for_user(
                message.chat_id, message.from_user.id, chain
            )

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
        persona_aliases: Optional[List[str]] = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        style_samples = [StyleSample(text=sample) for sample in samples]
        return await loop.run_in_executor(
            None,
            lambda: self._style.generate_reply(
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
                persona_aliases,
            ),
        )

    async def _generate_dialogue(
        self,
        username_a: str,
        persona_name_a: str,
        samples_a: List[str],
        style_summary_a: Optional[str],
        persona_card_a: Optional[str],
        relationship_hint_a: Optional[str],
        aliases_a: Optional[List[str]],
        username_b: str,
        persona_name_b: str,
        samples_b: List[str],
        style_summary_b: Optional[str],
        persona_card_b: Optional[str],
        relationship_hint_b: Optional[str],
        aliases_b: Optional[List[str]],
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
            aliases=aliases_a,
        )
        participant_b = DialogueParticipant(
            username=username_b,
            name=persona_name_b,
            samples=[StyleSample(text=sample) for sample in samples_b],
            style_summary=style_summary_b,
            persona_card=persona_card_b,
            relationship_hint=relationship_hint_b,
            aliases=aliases_b,
        )
        return await loop.run_in_executor(
            None,
            self._style.generate_dialogue,
            participant_a,
            participant_b,
            topic,
        )

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

    @property
    def recent_targets(self) -> Dict[Tuple[int, int], int]:
        return self._recent_targets

    def remember_target(self, chat_id: int, user_id: int, persona_id: int) -> None:
        self._recent_targets[(chat_id, user_id)] = persona_id

    def get_recent_target(self, chat_id: int, user_id: int) -> Optional[int]:
        return self._recent_targets.get((chat_id, user_id))

    async def handle_chain(
        self, message: Message, user_row: sqlite3.Row, chain: ImitationChain
    ) -> None:
        await self._handle_imitation_for_user(message, user_row, chain)

    def get_chain_for_message(
        self, chat_id: int, message_id: int
    ) -> Optional[ImitationChain]:
        return self._imitation.get_chain_for_message(chat_id, message_id)

    async def maybe_handle_direct_imitation(
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
        resolved_row, _ = await self._resolve_user_descriptor(chat.id, descriptor)
        if resolved_row is None:
            return False
        payload = remainder or self._imitation.extract_payload(
            message, cleaned_instruction
        )
        if not payload:
            payload = self._imitation.extract_payload(message, stripped)
        if not payload:
            payload = self._imitation.extract_reply_text(message)
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
            user_text = payload or "Продолжи разговор."
        chain = self._imitation.create_chain(
            chat.id,
            resolved_row,
            message.from_user,
            user_text,
            context_messages=self._imitation.collect_initial_context(message),
        )
        await self.handle_chain(message, resolved_row, chain)
        return True

    async def maybe_handle_followup(
        self,
        message: Message,
        cleaned_instruction: str,
        cleaned_lower: str,
        stripped: str,
        base_chain: Optional[ImitationChain],
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
        elif reply_to_bot and fallback_persona_id is not None:
            if message.chat_id is None:
                return False
            persona_row = await self._run_db(
                self._db.get_user_by_id, fallback_persona_id
            )
            if persona_row is None:
                return False
            chain_source = self._imitation.create_chain(
                message.chat_id,
                persona_row,
                message.from_user,
                "",
                context_messages=self._imitation.collect_initial_context(message),
            )
        else:
            return False

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
        if not reply_to_bot and not any(marker in cleaned_lower for marker in followup_markers):
            return False

        persona_name = chain_source.persona_name
        payload = self._imitation.extract_payload(
            message,
            cleaned_instruction,
            keywords=["имитируй", "ответ", "скажи", "соглас"],
        )
        user_text = self._imitation.prepare_chain_user_text(
            instruction=cleaned_instruction,
            payload=payload,
            descriptor=None,
            persona_row=persona_row,
            persona_name=persona_name,
            message=message,
        )
        if not user_text and message.text:
            stripped_original = self._imitation.strip_call_signs(message.text)
            stripped_original = self._imitation.normalize_chain_text(stripped_original)
            if stripped_original:
                user_text = stripped_original
        if not user_text:
            reply_text = self._imitation.extract_reply_text(message)
            if reply_text:
                user_text = self._imitation.normalize_chain_text(reply_text)
        if not user_text:
            return False

        chain = self._imitation.branch_chain(chain_source, message.from_user, user_text)
        await self.handle_chain(message, persona_row, chain)
        return True

    @staticmethod
    def _extract_leading_descriptor(text: str) -> Tuple[Optional[str], Optional[str]]:
        cleaned = text.strip()
        if not cleaned:
            return None, None
        cleaned = cleaned.lstrip(",:;—- ")
        if not cleaned:
            return None, None
        match = re.match(r"^(@?[\w\-]+)", cleaned)
        if not match:
            return None, None
        descriptor = match.group(1)
        remainder = cleaned[match.end() :].lstrip(" ,:;—-")
        return descriptor, (remainder or None)
