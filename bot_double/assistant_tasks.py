from __future__ import annotations

from dataclasses import dataclass
from openai import OpenAI


@dataclass(slots=True)
class TaskResult:
    text: str


class AssistantTaskEngine:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def translate(self, text: str, target_language: str) -> TaskResult:
        prompt = (
            "Ты — профессиональный переводчик. Переведи следующий текст на "
            f"{target_language} без добавления комментариев, пояснений или отметок."
            " Сохрани смысл, стиль и форматирование.\n\n"
            "Текст:\n"
            f"{text}\n\n"
            "Выведи только перевод."
        )
        return self._run(prompt)

    def summarize(self, text: str) -> TaskResult:
        prompt = (
            "Сделай краткое, но информативное резюме текста ниже."
            " Выдели основные тезисы и итог. Не превышай 3-4 предложения."
            "\n\nТекст:\n"
            f"{text}\n\n"
            "Ответ:"
        )
        return self._run(prompt)

    def paraphrase(self, text: str, instruction: str) -> TaskResult:
        prompt = (
            "Перепиши текст ниже, следуя инструкции. Сохрани факты и ключевой смысл."
            "\n\nИнструкция:"
            f" {instruction}\n\nТекст:\n{text}\n\nОтвет:"
        )
        return self._run(prompt)

    def proofread(self, text: str) -> TaskResult:
        prompt = (
            "Проверь текст на орфографические, пунктуационные и стилистические ошибки."
            " Исправь их и выведи отредактированную версию без пояснений.\n\nТекст:\n"
            f"{text}\n\nИсправленный текст:"
        )
        return self._run(prompt)

    def listify(self, text: str) -> TaskResult:
        prompt = (
            "Преобразуй текст в чёткий список задач."
            " Каждую задачу начинай с маркера '-'."
            " Если возможно, укажи краткий результат для каждой."
            "\n\nТекст:\n"
            f"{text}\n\nСписок:"
        )
        return self._run(prompt)

    def respond_helpfully(self, text: str, instruction: str) -> TaskResult:
        prompt = (
            "Сформулируй ответ на сообщение ниже."
            f" Следуй инструкции: {instruction}."
            " Сохрани тон дружелюбным и естественным.\n\nСообщение:\n"
            f"{text}\n\nОтвет:"
        )
        return self._run(prompt)

    def _run(self, prompt: str) -> TaskResult:
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
        )
        text = getattr(response, "output_text", "")
        return TaskResult(text=text.strip())
