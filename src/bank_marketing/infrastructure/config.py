"""Lecture des fichiers de configuration YAML."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Charge un fichier de configuration YAML.

    Parameters
    ----------
    path : Path
        Chemin du fichier YAML.

    Returns
    -------
    dict[str, Any]
        Contenu du fichier sous forme de dictionnaire.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    logger.info("Configuration chargée depuis %s", path)
    return config


def save_config(config: dict[str, Any], path: Path) -> None:
    """Enregistre une configuration au format YAML.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration à enregistrer.
    path : Path
        Chemin du fichier YAML de destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False, allow_unicode=True)
    logger.info("Configuration enregistrée dans %s", path)
