import abc
import logging
from contextlib import contextmanager

import attr


@attr.s(frozen=True, auto_attribs=True)
class LoggingLevel:
    python: str
    catboost: str


LOGGING_LEVELS = {
    "silent": LoggingLevel(python="NOTSET", catboost="Silent"),
    "debug": LoggingLevel(python="DEBUG", catboost="Debug"),
    "info": LoggingLevel(python="INFO", catboost="Info"),
    "stats": LoggingLevel(python="INFO", catboost="Verbose"),
    "error": LoggingLevel(python="ERROR", catboost="Silent"),
}


class LoggedClass(abc.ABC):
    def __init__(self, logging_level: str = "stats") -> None:
        # print(
        #     self.__class__.__name__,
        #     logging_level.lower(),
        #     LOGGING_LEVELS[logging_level.lower()].python,
        # )
        self._logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s] [%(filename)s:%(lineno)d] > %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.setLevel(LOGGING_LEVELS[logging_level.lower()].python)
        self._logger.addHandler(handler)
        self._logger.setLevel(LOGGING_LEVELS[logging_level.lower()].python)


@contextmanager
def create_logger(logging_level: str = "stats") -> logging.Logger:
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s.%(funcName)s:%(lineno)d] > %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(LOGGING_LEVELS[logging_level.lower()].python)
    logger.addHandler(handler)
    logger.setLevel(LOGGING_LEVELS[logging_level.lower()].python)
    yield logger
    logger.removeHandler(handler)
