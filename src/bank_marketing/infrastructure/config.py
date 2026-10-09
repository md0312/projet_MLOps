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
