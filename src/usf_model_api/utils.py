import logging

logging.basicConfig()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and set its level to INFO.
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    return log
