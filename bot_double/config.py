from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    bot_token: str
    openai_api_key: str
    db_path: Path
    openai_model: str = "gpt-5-nano"
    openai_reasoning_effort: Optional[str] = None
    openai_text_verbosity: Optional[str] = None
    auto_imitate_probability: float = 0.2
    min_messages_for_profile: int = 20
    max_messages_per_user: int = 200
    prompt_samples: int = 30
    dialog_context_messages: int = 6
    style_recent_messages: int = 5
    min_tokens_to_store: int = 3
    peer_profile_count: int = 3
    peer_profile_samples: int = 2
    burst_inactivity_seconds: int = 10
    burst_gap_seconds: int = 12
    burst_max_duration_seconds: int = 90
    burst_max_parts: int = 6
    burst_max_chars: int = 2000
    turn_window_seconds: int = 10
    enable_bursts: bool = True
    enable_voice_transcription: bool = False
    voice_transcription_model: str = "gpt-4o-mini-transcribe"
    voice_transcription_language: Optional[str] = "ru"
    voice_transcription_max_duration: int = 180
    relationship_analysis_model: Optional[str] = None
    relationship_analysis_min_pending: int = 5
    relationship_analysis_min_hours: int = 24
    persona_analysis_model: Optional[str] = None
    persona_analysis_min_messages: int = 50
    persona_analysis_max_messages: int = 100
    persona_analysis_min_hours: int = 24


class SettingsError(RuntimeError):
    pass


def _get_env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise SettingsError(f"Environment variable {name} must be a float") from exc


def _get_env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:  # pragma: no cover - defensive
            raise SettingsError(f"Environment variable {name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise SettingsError(f"Environment variable {name} must be >= {minimum}")
    return value


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"Environment variable {name} must be a boolean value")


def load_settings() -> Settings:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise SettingsError("TELEGRAM_BOT_TOKEN is not set")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise SettingsError("OPENAI_API_KEY is not set")

    db_path = Path(os.getenv("BOT_DOUBLE_DB_PATH", "bot_double.db"))
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT")
    if reasoning_effort:
        allowed_efforts = {"minimal", "low", "medium", "high"}
        if reasoning_effort not in allowed_efforts:
            raise SettingsError(
                "OPENAI_REASONING_EFFORT must be one of minimal, low, medium, high"
            )
    else:
        reasoning_effort = None

    text_verbosity = os.getenv("OPENAI_TEXT_VERBOSITY")
    if text_verbosity:
        allowed_verbosity = {"low", "medium", "high"}
        if text_verbosity not in allowed_verbosity:
            raise SettingsError(
                "OPENAI_TEXT_VERBOSITY must be one of low, medium, high"
            )
    else:
        text_verbosity = None

    probability = _get_env_float("AUTO_IMITATE_PROBABILITY", 0.2) or 0.0
    min_messages = _get_env_int("MIN_MESSAGES_FOR_PROFILE", 20, minimum=1)
    max_messages = _get_env_int("MAX_MESSAGES_PER_USER", 200, minimum=min_messages)
    prompt_samples = _get_env_int("PROMPT_SAMPLE_SIZE", 30, minimum=1)
    dialog_context_messages = _get_env_int("DIALOG_CONTEXT_MESSAGES", 6, minimum=0)
    style_recent_messages = _get_env_int("STYLE_RECENT_MESSAGES", 5, minimum=0)
    peer_profile_count = _get_env_int("PEER_PROFILE_COUNT", 3, minimum=0)
    peer_profile_samples = _get_env_int("PEER_PROFILE_SAMPLES", 2, minimum=0)
    min_tokens_to_store = _get_env_int("MIN_TOKENS_TO_STORE", 3, minimum=1)
    short_buffer_seconds = _get_env_int("SHORT_MESSAGE_BUFFER_SECONDS", 10, minimum=1)
    burst_inactivity_seconds = _get_env_int(
        "BURST_INACTIVITY_SECONDS", short_buffer_seconds, minimum=1
    )
    burst_gap_seconds = _get_env_int("BURST_GAP_SECONDS", 12, minimum=1)
    burst_max_duration_seconds = _get_env_int(
        "BURST_MAX_DURATION_SECONDS", 90, minimum=0
    )
    burst_max_parts = _get_env_int("BURST_MAX_PARTS", 6, minimum=1)
    burst_max_chars = _get_env_int("BURST_MAX_CHARS", 2000, minimum=0)
    turn_window_seconds = _get_env_int("TURN_WINDOW_SECONDS", 10, minimum=1)
    enable_bursts = _get_env_bool("ENABLE_BURSTS", True)
    enable_voice_transcription = _get_env_bool("ENABLE_VOICE_TRANSCRIPTION", False)
    voice_transcription_model = os.getenv(
        "VOICE_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
    )
    voice_transcription_language = os.getenv("VOICE_TRANSCRIPTION_LANGUAGE", "ru")
    if voice_transcription_language == "":
        voice_transcription_language = None
    voice_transcription_max_duration = _get_env_int(
        "VOICE_TRANSCRIPTION_MAX_DURATION", 180, minimum=0
    )
    rel_model = os.getenv("RELATIONSHIP_ANALYSIS_MODEL")
    rel_min_pending = _get_env_int("RELATIONSHIP_ANALYSIS_MIN_PENDING", 5, minimum=1)
    rel_min_hours = _get_env_int("RELATIONSHIP_ANALYSIS_MIN_HOURS", 24, minimum=0)
    persona_model = os.getenv("PERSONA_ANALYSIS_MODEL")
    persona_min_msgs = _get_env_int(
        "PERSONA_ANALYSIS_MIN_MESSAGES", 50, minimum=10
    )
    persona_max_msgs = _get_env_int(
        "PERSONA_ANALYSIS_MAX_MESSAGES", 100, minimum=persona_min_msgs
    )
    persona_min_hours = _get_env_int("PERSONA_ANALYSIS_MIN_HOURS", 24, minimum=0)

    if rel_model == "":
        rel_model = None
    if persona_model == "":
        persona_model = None
    if rel_model is None:
        rel_model = openai_model
    if persona_model is None:
        persona_model = openai_model

    return Settings(
        bot_token=bot_token,
        openai_api_key=openai_api_key,
        db_path=db_path,
        openai_model=openai_model,
        openai_reasoning_effort=reasoning_effort,
        openai_text_verbosity=text_verbosity,
        auto_imitate_probability=probability,
        min_messages_for_profile=min_messages,
        max_messages_per_user=max_messages,
        prompt_samples=prompt_samples,
        dialog_context_messages=dialog_context_messages,
        style_recent_messages=style_recent_messages,
        min_tokens_to_store=min_tokens_to_store,
        peer_profile_count=peer_profile_count,
        peer_profile_samples=peer_profile_samples,
        burst_inactivity_seconds=burst_inactivity_seconds,
        burst_gap_seconds=burst_gap_seconds,
        burst_max_duration_seconds=burst_max_duration_seconds,
        burst_max_parts=burst_max_parts,
        burst_max_chars=burst_max_chars,
        turn_window_seconds=turn_window_seconds,
        enable_bursts=enable_bursts,
        enable_voice_transcription=enable_voice_transcription,
        voice_transcription_model=voice_transcription_model,
        voice_transcription_language=voice_transcription_language,
        voice_transcription_max_duration=voice_transcription_max_duration,
        relationship_analysis_model=rel_model,
        relationship_analysis_min_pending=rel_min_pending,
        relationship_analysis_min_hours=rel_min_hours,
        persona_analysis_model=persona_model,
        persona_analysis_min_messages=persona_min_msgs,
        persona_analysis_max_messages=persona_max_msgs,
        persona_analysis_min_hours=persona_min_hours,
    )
