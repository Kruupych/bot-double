from __future__ import annotations

import asyncio
import logging
import random
from functools import partial
from typing import Any, Callable, List, Optional, Set, TypeVar

from telegram import Message, MessageEntity, Update
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
from .style_engine import ContextMessage, StyleEngine, StyleSample
from .utils import display_name, should_store_message

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class BotDouble:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = Database(settings.db_path, settings.max_messages_per_user)
        self._style = StyleEngine(settings.openai_api_key, model=settings.openai_model)
        self._bot_id: Optional[int] = None
        self._bot_name: str = "Бот-Двойник"

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

        samples = await self._collect_style_samples(chat.id, int(user_row["id"]))
        if not samples:
            await message.reply_text(
                f"Пока не могу имитировать @{username}, сообщений недостаточно."
            )
            return

        persona_name = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        context_messages = await self._get_dialog_context(chat.id)
        try:
            ai_reply = await self._generate_reply(
                username, persona_name, samples, starter, context_messages
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
        samples = await self._collect_style_samples(chat.id, int(user_row["id"]))
        if not samples:
            return

        persona_name = display_name(
            user_row["username"], user_row["first_name"], user_row["last_name"]
        )
        context_messages = await self._get_dialog_context(chat.id)
        try:
            ai_reply = await self._generate_reply(
                username, persona_name, samples, starter, context_messages
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

    # --- internal helpers -----------------------------------------------------------
    async def _capture_message(self, message: Message) -> None:
        if not should_store_message(
            message,
            min_tokens=self._settings.min_tokens_to_store,
            allowed_bot_id=self._bot_id,
        ):
            return
        if message.from_user is None:
            return
        user = message.from_user
        user_id = await self._run_db(
            self._db.upsert_user,
            user.id,
            user.username,
            user.first_name,
            user.last_name,
        )
        timestamp = int(message.date.timestamp())
        await self._run_db(
            self._db.store_message,
            message.chat_id,
            user_id,
            message.text or "",
            timestamp,
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

    async def _generate_reply(
        self,
        username: str,
        persona_name: str,
        samples: List[str],
        starter: str,
        context_messages: Optional[List[ContextMessage]],
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
        )

    async def _collect_style_samples(self, chat_id: int, user_id: int) -> List[str]:
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

        combined: List[str] = []
        seen: Set[str] = set()

        for text in recent_messages + random_pool:
            normalized = text.strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            combined.append(normalized)
            seen.add(normalized)
            if len(combined) >= total_target:
                break

        return combined

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
    application.add_handler(CommandHandler("forgetme", bot.forget_me))
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, bot.on_new_members))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_text_message))

    application.run_polling(close_loop=False)
