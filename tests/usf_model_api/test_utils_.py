# pylint: disable=unused-argument
import logging
from unittest.mock import patch, mock_open
import pytest
import yaml

from usf_model_api.utils import get_logger, load_yaml


def test_get_logger():
    logger = get_logger("test_logger", logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG


@patch("builtins.open", new_callable=mock_open, read_data="catboost:\n  verbose: 100\n")
def test_load_yaml_success(mock_file):
    file_path = "test.yaml"
    with patch("yaml.safe_load", return_value={"catboost": {"verbose": 100}}):
        data = load_yaml(file_path)
        assert data == {"catboost": {"verbose": 100}}
        mock_file.assert_called_once_with(file_path, "r")


@patch("builtins.open", new_callable=mock_open)
def test_load_yaml_file_not_found(mock_file):
    file_path = "non_existent.yaml"
    mock_file.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_yaml(file_path)


@patch("builtins.open", new_callable=mock_open, read_data="invalid_yaml")
def test_load_yaml_parsing_error(mock_file):
    file_path = "invalid.yaml"
    with patch("yaml.safe_load", side_effect=yaml.YAMLError):
        with pytest.raises(yaml.YAMLError):
            load_yaml(file_path)
