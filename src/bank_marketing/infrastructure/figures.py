"""Sauvegarde des figures retenues pour le rapport."""

import logging
from pathlib import Path

from matplotlib.figure import Figure

from bank_marketing.settings import FIGURES_DIR

logger = logging.getLogger(__name__)


def save_figure(figure: Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Enregistre une figure matplotlib au format PNG.

    Parameters
    ----------
    figure : Figure
        Figure à enregistrer.
    name : str
        Nom du fichier, sans extension (par exemple ``"distribution_cible"``).
    directory : Path, optional
        Dossier de destination. Par défaut, ``reports/figures``.

    Returns
    -------
    Path
        Chemin du fichier créé.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Figure enregistrée dans %s", path)
    return path
