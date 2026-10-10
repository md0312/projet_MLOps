"""Outils communs pour le suivi des expériences avec MLflow."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import skops.io
from sklearn.base import BaseEstimator

from bank_marketing.settings import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)


def setup_mlflow(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    artifacts_dir: Path = MLFLOW_ARTIFACTS_DIR,
) -> str:
    """Configure MLflow et active l'expérience du projet.

    Les runs sont enregistrés dans une base SQLite et leurs artefacts dans un
    dossier à la racine du projet, quel que soit le dossier d'exécution.

    Parameters
    ----------
    experiment_name : str, optional
        Nom de l'expérience MLflow.
    tracking_uri : str, optional
        Adresse du stockage des runs (par défaut ``sqlite:///.../mlflow.db``).
    artifacts_dir : Path, optional
        Dossier des artefacts (modèles, fichiers de résultats).

    Returns
    -------
    str
        Identifiant de l'expérience active.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifacts_dir.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    logger.info("Expérience MLflow active : %s (%s)", experiment_name, tracking_uri)
    return experiment_id


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplatit une configuration imbriquée en paramètres MLflow.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration, éventuellement imbriquée.
    prefix : str, optional
        Préfixe ajouté aux clés (utilisé lors de la récursion).

    Returns
    -------
    dict[str, Any]
        Dictionnaire à un seul niveau, dont les clés sont reliées par des
        points (par exemple ``model.params.C``).

    Examples
    --------
    >>> flatten_config({"model": {"name": "svm", "params": {"C": 1.0}}})
    {'model.name': 'svm', 'model.params.C': 1.0}
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=f"{full_key}."))
        else:
            flat[full_key] = value
    return flat


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Préfixe le nom des métriques pour distinguer leur origine.

    Parameters
    ----------
    metrics : dict[str, float]
        Métriques à renommer.
    prefix : str
        Préfixe, par exemple ``"test"`` pour obtenir ``test_f1``.

    Returns
    -------
    dict[str, float]
        Métriques renommées.
    """
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def list_types_to_trust(model: BaseEstimator) -> list[str]:
    """Liste les types à déclarer sûrs pour enregistrer un modèle dans MLflow.

    MLflow sérialise les modèles scikit-learn avec skops, qui refuse par défaut
    les types qu'il ne connaît pas (types NumPy, calibrage du SVM, etc.). Le
    modèle étant produit par le code du projet, ses types sont sûrs.

    Parameters
    ----------
    model : BaseEstimator
        Modèle ou pipeline entraîné par le projet.

    Returns
    -------
    list[str]
        Noms complets des types à transmettre à ``skops_trusted_types``.
    """
    return skops.io.get_untrusted_types(data=skops.io.dumps(model))
