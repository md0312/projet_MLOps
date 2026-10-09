"""Sauvegarde et chargement du modèle entraîné et de son seuil de décision."""

import logging
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def save_model(model: BaseEstimator, threshold: float, path: Path) -> None:
    """Enregistre le modèle et son seuil de décision dans un même fichier.

    Le seuil est conservé avec le modèle : une prédiction n'a de sens qu'avec le
    seuil choisi lors de l'entraînement.

    Parameters
    ----------
    model : BaseEstimator
        Pipeline scikit-learn entraîné (prétraitement et modèle).
    threshold : float
        Seuil de probabilité au-delà duquel un client est prédit souscripteur.
    path : Path
        Chemin du fichier ``.joblib`` de destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, path)
    logger.info("Modèle enregistré dans %s (seuil = %.3f)", path, threshold)


def load_model(path: Path) -> tuple[BaseEstimator, float]:
    """Charge un modèle et son seuil de décision.

    Parameters
    ----------
    path : Path
        Chemin du fichier ``.joblib`` créé par :func:`save_model`.

    Returns
    -------
    tuple[BaseEstimator, float]
        Le pipeline entraîné et son seuil de décision.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    artifact = joblib.load(path)
    logger.info("Modèle chargé depuis %s", path)
    return artifact["model"], artifact["threshold"]
