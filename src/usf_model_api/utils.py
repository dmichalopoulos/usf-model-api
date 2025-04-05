import logging
from typing import Dict, Any
from pathlib import Path
import yaml

logging.basicConfig()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and set its level to the specified logging level.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level to set for the logger (default is logging.INFO).

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    return log


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the YAML file.

    Returns
    -------
    dict
        The contents of the YAML file as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML file.
    """
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        return data
    except FileNotFoundError as e:
        logging.error("The file %s was not found.", file_path)
        raise e
    except yaml.YAMLError as e:
        logging.error("Error parsing YAML file %s: %s", file_path, e)
        raise e
