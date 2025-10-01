from __future__ import annotations

from typing import Awaitable, Callable, Optional, Sequence, TypeVar

from telegram import Update, User
from telegram.ext import ContextTypes
from .db import Database
from .utils import display_name


T = TypeVar("T")
RunDB = Callable[..., Awaitable[T]]


class CommandService:
    """Handles user-facing command logic for BotDouble."""

    def __init__(
        self,
        *,
        db: Database,
        run_db: RunDB,
        invalidate_alias_cache: Callable[[Optional[int]], None],
        get_persona_card: Callable[[int, int], Awaitable[Optional[str]]],
        get_relationship_summary_text: Callable[[int, int, int], Awaitable[Optional[str]]],
        ensure_internal_user: Callable[[Optional[User]], Awaitable[Optional[int]]],
    ) -> None:
        self._db = db
        self._run_db = run_db
        self._invalidate_alias_cache = invalidate_alias_cache
        self._get_persona_card = get_persona_card
        self._get_relationship_summary_text = get_relationship_summary_text
        self._ensure_internal_user = ensure_internal_user

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
            await message.reply_text("Использование: /persona_mode @username <card|summary|auto>")
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
        mode_map = {"summary": 0, "card": 1, "auto": 2}
        if mode_token not in mode_map:
            await message.reply_text("Режим должен быть одним из: card, summary, auto")
            return
        await self._run_db(
            self._db.set_persona_preference, chat.id, internal_id, mode_map[mode_token]
        )
        await message.reply_text(f"Режим для @{username}: {mode_token}.")

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
        response_lines = [f"Профиль {target_name}:"]
        if persona_card:
            response_lines.append(persona_card)
        else:
            response_lines.append("Карточка персоны ещё не готова. Продолжайте общаться!")

        relationship_lines = []
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
                relationship_lines.append(f"Отношение к {other_name}: {summary}")
            else:
                relationship_lines.append(
                    f"Отношение к {other_name}: данных пока мало."
                )
            reverse = await self._get_relationship_summary_text(
                chat.id, relationship_target_internal_id, target_internal_id
            )
            if reverse:
                relationship_lines.append(f"Ответная позиция {other_name}: {reverse}")
        elif target_internal_id != requester_internal_id:
            summary = await self._get_relationship_summary_text(
                chat.id, requester_internal_id, target_internal_id
            )
            if summary:
                response_lines.append("")
                response_lines.append(f"Вы о {target_name}: {summary}")
            reverse = await self._get_relationship_summary_text(
                chat.id, target_internal_id, requester_internal_id
            )
            if reverse:
                response_lines.append(f"{target_name} о вас: {reverse}")

        if relationship_lines:
            response_lines.append("")
            response_lines.extend(relationship_lines)

        text = "\n".join(line for line in response_lines if line is not None)
        await message.reply_text(text or "Нет данных", disable_web_page_preview=True)


def _split_aliases(raw_section: str) -> Sequence[str]:
    aliases = [alias.strip() for alias in raw_section.split(";")]
    aliases = [chunk for alias in aliases for chunk in alias.split(",")]
    return [alias.strip() for alias in aliases if alias.strip()]
