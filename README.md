# Бот-Двойник

Telegram-бот, который автоматически собирает сообщения участников чата, формирует актуальные "слепки стиля" и по запросу имитирует выбранного пользователя с помощью модели OpenAI `gpt-5-nano`.

## Возможности
- Автоматический сбор сообщений и хранение последних N фраз для каждого участника чата.
- Имитация участника по команде `/imitate @username [текст-затравка]`.
- Автоответы в стиле упомянутого пользователя (при включённом режиме автоимитации).
- Команда `/imitate_profiles` показывает статус собранных профилей.

## Быстрый старт
1. Создайте файл `.env` или задайте переменные окружения:
   ```env
   TELEGRAM_BOT_TOKEN=...  # токен бота от BotFather
   OPENAI_API_KEY=...      # ключ OpenAI
   ```
   Дополнительно можно настроить:
   ```env
   BOT_DOUBLE_DB_PATH=bot_double.db
   AUTO_IMITATE_PROBABILITY=0.25
   MIN_MESSAGES_FOR_PROFILE=20
   MAX_MESSAGES_PER_USER=200
   PROMPT_SAMPLE_SIZE=30
   DIALOG_CONTEXT_MESSAGES=6
   STYLE_RECENT_MESSAGES=5
   PEER_PROFILE_COUNT=3
   PEER_PROFILE_SAMPLES=2
   OPENAI_MODEL=gpt-5-nano
   OPENAI_REASONING_EFFORT=medium
   OPENAI_TEXT_VERBOSITY=medium
   MIN_TOKENS_TO_STORE=3
   ```

2. Установите зависимости:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Запустите бота:
   ```bash
   python main.py
   ```

## Команды
- `/imitate @username [текст]` — сгенерировать ответ в стиле пользователя.
- `/imitate_profiles` — показать, чьи профили готовы.
- `/auto_imitate_on` / `/auto_imitate_off` — включить или отключить автопародию в чате.
- `/dialogue @user1 @user2 [тема]` — разыграть мини-диалог между двумя участниками в их стиле.
- `/forgetme` — удалить все сохранённые сообщения и профиль пользователя.

Бот также периодически переанализирует пары собеседников, чтобы лучше понимать их отношения и тон общения. Это улучшает контекст в имитациях и диалогах.

## Хранение данных
Все данные сохраняются в SQLite (по умолчанию `bot_double.db`). Для каждого пользователя бот хранит скользящее окно из последних `MAX_MESSAGES_PER_USER` сообщений. При превышении лимита старые записи автоматически удаляются.
