"""Configuration centralisée du module logging."""

import logging.config
from pathlib import Path

import yaml

from bank_marketing.settings import LOGGING_CONFIG_PATH


def configure_logging(config_path: Path = LOGGING_CONFIG_PATH) -> None:
    """Configure le logging à partir d'un fichier YAML.

    Parameters
    ----------
    config_path : Path, optional
        Chemin du fichier de configuration du logging. Par défaut, le fichier
        ``logging.yaml`` du module ``settings``.
    """
    with config_path.open(encoding="utf-8") as config_file:
        logging.config.dictConfig(yaml.safe_load(config_file))
