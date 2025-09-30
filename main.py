from __future__ import annotations

import asyncio
import logging
import sys

from bot_double.bot import run_bot
from bot_double.config import SettingsError, load_settings


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    try:
        settings = load_settings()
    except SettingsError as exc:
        logging.getLogger(__name__).error("Configuration error: %s", exc)
        sys.exit(1)

    run_bot(settings)


if __name__ == "__main__":
    main()

