import logging

from bank_marketing import settings
from bank_marketing.settings.logging_setup import configure_logging


def test_project_root_contains_pyproject():
    assert (settings.PROJECT_ROOT / "pyproject.toml").is_file()


def test_raw_data_file_exists():
    assert settings.RAW_DATA_PATH.is_file()


def test_logging_configuration_file_exists():
    assert settings.LOGGING_CONFIG_PATH.is_file()


def test_configure_logging_sets_info_level():
    configure_logging()
    assert logging.getLogger().level == logging.INFO
