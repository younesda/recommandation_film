import logging
import sys


_LOGGING_INITIALIZED = False


def configure_logging(level: str = "INFO") -> None:
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
