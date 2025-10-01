from __future__ import annotations

import io
from typing import Optional

from openai import OpenAI


class SpeechTranscriber:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._language = language
        self._prompt = prompt
        self._temperature = temperature

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        mime_type: Optional[str] = None,
        *,
        language: Optional[str] = None,
    ) -> Optional[str]:
        buffer = io.BytesIO(audio_bytes)
        if "." in filename:
            buffer.name = filename
        elif mime_type and "/" in mime_type:
            extension = mime_type.split("/")[-1]
            buffer.name = f"{filename}.{extension}"
        else:
            buffer.name = filename
        options: dict[str, object] = {
            "model": self._model,
            "file": buffer,
            "response_format": "text",
        }
        effective_language = language or self._language
        if effective_language:
            options["language"] = effective_language
        if self._prompt:
            options["prompt"] = self._prompt
        if self._temperature is not None:
            options["temperature"] = self._temperature
        response = self._client.audio.transcriptions.create(**options)
        if isinstance(response, str):
            return response.strip() or None
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text.strip() or None
        return None
