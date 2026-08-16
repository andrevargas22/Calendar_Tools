"""
Logging configuration utilities.

Same pattern as Colorado_IA/src/utils/logging_config.py: timestamps forced to
America/Sao_Paulo regardless of host timezone, plain stdout stream handler.
"""

import datetime
import logging
import sys
import zoneinfo
from typing import Optional

_TZ = zoneinfo.ZoneInfo("America/Sao_Paulo")


class _LocalTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging with consistent formatting."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_LocalTimeFormatter(format_string))
    logger.addHandler(handler)

    return logger
