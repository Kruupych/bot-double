from __future__ import annotations

from typing import Awaitable, Callable, Optional, Sequence, TypeVar

from telegram import Update, User
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from .config import Settings
from .db import Database
from .utils import display_name


T = TypeVar("T")
RunDB = Callable[..., Awaitable[T]]


class CommandService:
    """Handles user-facing command logic for BotDouble."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        run_db: RunDB,
        invalidate_alias_cache: Callable[[Optional[int]], None],
        get_persona_card: Callable[[int, int], Awaitable[Optional[str]]],
        get_style_summary: Callable[[int, int], Awaitable[Optional[str]]],
        get_relationship_summary_text: Callable[[int, int, int], Awaitable[Optional[str]]],
        ensure_internal_user: Callable[[Optional[User]], Awaitable[Optional[int]]],
        flush_buffers_for_chat: Callable[[int], Awaitable[None]],
    ) -> None:
        self._settings = settings
        self._db = db
        self._run_db = run_db
        self._invalidate_alias_cache = invalidate_alias_cache
        self._get_persona_card = get_persona_card
        self._get_style_summary = get_style_summary
        self._get_relationship_summary_text = get_relationship_summary_text
        self._ensure_internal_user = ensure_internal_user
        self._flush_buffers_for_chat = flush_buffers_for_chat

    async def alias_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        text = message.text or ""
        parts = text.strip().split(None, 2)
        if len(parts) < 2:
            await message.reply_text("Использование: /alias @username прозвище1, прозвище2")
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
            await message.reply_text("Добавьте хотя бы одно прозвище после @username.")
            return
        aliases = _split_aliases(alias_section)
        if not aliases:
            await message.reply_text("Не удалось найти прозвища. Разделяйте их запятыми.")
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
        lines = []
        if added:
            lines.append("Добавил прозвища: " + ", ".join(list(dict.fromkeys(added))))
        if skipped:
            lines.append("Пропущено (уже есть или пустые): " + ", ".join(list(dict.fromkeys(skipped))))
        if not lines:
            lines.append("Ничего не добавлено.")
        await message.reply_text("\n".join(lines))

    async def alias_reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
            await message.reply_text(f"Удалено {deleted} прозвищ для @{username}.")
        else:
            await message.reply_text(f"Для @{username} не было сохранённых прозвищ.")

    async def persona_mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        if len(context.args) < 2:
            await message.reply_text(
                "Использование: /persona_mode @username <card|summary|combined|auto>"
            )
            return
        username_token, mode_token = context.args[0], context.args[1].lower().strip()
        if not username_token.startswith("@"):
            await message.reply_text("Первым аргументом должен быть @username")
            return
        username = username_token.lstrip("@")
        row = await self._run_db(self._db.get_user_by_username, username)
        if row is None:
            await message.reply_text(f"Я ещё не знаю пользователя @{username}.")
            return
        internal_id = int(row["id"])
        mode_map = {"summary": 0, "card": 1, "auto": 2, "combined": 3}
        if mode_token not in mode_map:
            await message.reply_text(
                "Режим должен быть одним из: card, summary, combined, auto"
            )
            return
        await self._run_db(
            self._db.set_persona_preference, chat.id, internal_id, mode_map[mode_token]
        )
        await message.reply_text(f"Режим для @{username}: {mode_token}.")

    async def imitate_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat:
            return
        await self._flush_buffers_for_chat(chat.id)

        lines = ["Статус профилей:"]
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
            "📝 Имитация стиля:",
            "• /imitate @username текст — ответить в стиле пользователя.",
            "• Упомяни меня + имя: '@бот, что скажет Антон про это?'",
            "• По прозвищу: 'двойник, ответь как Тоха' (прозвища через /alias).",
            "• В реплае на мой ответ: 'продолжи', 'согласись', 'добавь деталей'.",
            "",
            "🎭 Развлечения:",
            "• /roast @user — жёсткая прожарка по стилю общения.",
            "• /horoscope @user — персональный гороскоп по поведению.",
            "• /tinder @user — генерация Tinder-профиля.",
            "• /compatibility @user1 @user2 — тест совместимости.",
            "• /battle @user1 @user2 — рэп-баттл между пользователями.",
            "• /dialogue @user1 @user2 [тема] — диалог между пользователями.",
            "• /story @user1 @user2... [тема] — короткий рассказ (2-5 участников).",
            "• /long_story @user1 @user2... [тема] — развёрнутый рассказ.",
            "• /news — новости чата в стиле жёлтой прессы.",
            "• /summary — краткое резюме последних событий в чате.",
            "• /conspiracy — теория заговора про чат.",
            "",
            "⚙️ Управление:",
            "• /alias @user имя, кличка — добавить прозвища для пользователя.",
            "• /alias_reset @user — удалить все прозвища.",
            "• /profile или /me — посмотреть свой профиль.",
            "• /forgetme — удалить все свои данные.",
            "",
            "💡 Совет: чем больше сообщений я видел от человека, тем лучше имитация!",
        ]
        await message.reply_text("\n".join(lines), disable_web_page_preview=True)

    async def auto_imitate_toggle(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, *, enabled: bool
    ) -> None:
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

    async def profile_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if not message or not chat or message.from_user is None:
            return

        await self._flush_buffers_for_chat(chat.id)

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
                    await message.reply_text(f"Я ещё не знаю пользователя @{second_username}.")
                    return
                relationship_target_internal_id = int(second_row["id"])

        if target_row is None:
            await message.reply_text("Нет данных о выбранном пользователе.")
            return

        target_name = display_name(
            target_row["username"], target_row["first_name"], target_row["last_name"]
        )

        persona_card = await self._get_persona_card(chat.id, target_internal_id)
        style_summary = await self._get_style_summary(chat.id, target_internal_id)

        response_lines = [f"📇 Профиль {target_name}"]

        def add_section(title: str, body: Sequence[str]) -> None:
            response_lines.append("")
            response_lines.append(title)
            response_lines.extend(body if body else [])

        def indent_block(text: str) -> list[str]:
            lines = text.splitlines()
            if not lines:
                return []
            return [f"  {line}" if line else "" for line in lines]

        persona_body: Sequence[str]
        if persona_card:
            persona_body = indent_block(persona_card)
        else:
            persona_body = ["  Карточка ещё не готова — продолжайте общаться."]
        add_section("🧬 Карточка персоны", persona_body)

        summary_body: Sequence[str]
        if style_summary:
            summary_body = indent_block(style_summary)
        else:
            summary_body = ["  Данных пока недостаточно для алгоритмического анализа."]
        add_section("📊 Алгоритмический разбор", summary_body)

        relationship_entries: list[str] = []
        if relationship_target_internal_id is not None:
            summary = await self._get_relationship_summary_text(
                chat.id, target_internal_id, relationship_target_internal_id
            )
            other_row = await self._run_db(
                self._db.get_user_by_id, relationship_target_internal_id
            )
            other_name = (
                display_name(
                    other_row["username"],
                    other_row["first_name"],
                    other_row["last_name"],
                )
                if other_row
                else "неизвестный пользователь"
            )
            if summary:
                relationship_entries.append(f"  → {other_name}: {summary}")
            else:
                relationship_entries.append(f"  → {other_name}: данных пока мало.")
            reverse = await self._get_relationship_summary_text(
                chat.id, relationship_target_internal_id, target_internal_id
            )
            if reverse:
                relationship_entries.append(f"  ← {other_name}: {reverse}")
        elif target_internal_id != requester_internal_id:
            summary = await self._get_relationship_summary_text(
                chat.id, requester_internal_id, target_internal_id
            )
            if summary:
                relationship_entries.append(f"  Вы о {target_name}: {summary}")
            reverse = await self._get_relationship_summary_text(
                chat.id, target_internal_id, requester_internal_id
            )
            if reverse:
                relationship_entries.append(f"  {target_name} о вас: {reverse}")

        if relationship_entries:
            add_section("🤝 Отношения", relationship_entries)

        text = "\n".join(line for line in response_lines if line is not None)
        await message.reply_text(text or "Нет данных", disable_web_page_preview=True)


def _split_aliases(raw_section: str) -> Sequence[str]:
    aliases = [alias.strip() for alias in raw_section.split(";")]
    aliases = [chunk for alias in aliases for chunk in alias.split(",")]
    return [alias.strip() for alias in aliases if alias.strip()]
