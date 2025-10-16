import logging
import os
from config import LOG_FILE, LOG_LEVEL

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

LOG_COLORS = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if COLORAMA_AVAILABLE:
            color = LOG_COLORS.get(record.levelname, "")
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def get_logger(name: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  

    logger.setLevel(LOG_LEVEL.upper())

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(LOG_LEVEL.upper())
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL.upper())
    console_formatter = ColoredFormatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    test_logger = get_logger("test")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    test_logger.critical("Critical message")