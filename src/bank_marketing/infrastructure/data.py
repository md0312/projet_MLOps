"""Lecture et écriture des données tabulaires (fichiers CSV)."""

import logging
from pathlib import Path

import pandas as pd

from bank_marketing.settings import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Charge le jeu de données brut Bank Marketing.

    Parameters
    ----------
    path : Path, optional
        Chemin du fichier CSV brut, séparé par des points-virgules. Par défaut,
        ``data/raw/bank-additional.csv``.

    Returns
    -------
    pd.DataFrame
        Données brutes, sans aucune transformation.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")

    data = pd.read_csv(path, sep=";")
    logger.info("Données chargées depuis %s : %d lignes, %d colonnes", path, *data.shape)
    return data


def save_dataframe(data: pd.DataFrame, path: Path) -> None:
    """Enregistre un DataFrame au format CSV en créant le dossier si besoin.

    Parameters
    ----------
    data : pd.DataFrame
        Tableau à enregistrer (métriques, prédictions, etc.).
    path : Path
        Chemin du fichier CSV de destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, lineterminator="\n")
    logger.info("Tableau enregistré dans %s", path)
