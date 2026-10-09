"""Construction des modèles et prédictions out-of-fold.

Trois familles de modèles ont été comparées lors de l'exploration : régression
logistique, forêt aléatoire et SVM calibré. Toutes compensent le déséquilibre
de la cible avec ``class_weight="balanced"``, précisé dans la configuration.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from bank_marketing.domain.preprocessing import create_preprocessor

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("logistic_regression", "random_forest", "svm")


def build_estimator(model_name: str, params: dict[str, Any], random_state: int) -> BaseEstimator:
    """Instancie un classifieur à partir de son nom et de ses hyperparamètres.

    Parameters
    ----------
    model_name : {"logistic_regression", "random_forest", "svm"}
        Famille de modèle.
    params : dict[str, Any]
        Hyperparamètres transmis au classifieur scikit-learn.
    random_state : int
        Graine aléatoire, pour un entraînement reproductible.

    Returns
    -------
    BaseEstimator
        Classifieur non entraîné, capable de produire des probabilités.

    Raises
    ------
    ValueError
        Si le nom du modèle n'est pas pris en charge.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(**params, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(**params, random_state=random_state)
    if model_name == "svm":
        # SVC(probability=True) est déprécié : le calibrage sigmoïde (Platt)
        # transforme les scores du SVM en probabilités.
        return CalibratedClassifierCV(SVC(**params), method="sigmoid", ensemble=False)
    raise ValueError(f"Modèle inconnu : {model_name!r} (attendu : {', '.join(SUPPORTED_MODELS)})")


def build_pipeline(
    features: pd.DataFrame,
    model_name: str,
    params: dict[str, Any],
    random_state: int,
) -> Pipeline:
    """Assemble le prétraitement et le classifieur dans un pipeline unique.

    Parameters
    ----------
    features : pd.DataFrame
        Variables explicatives, utilisées pour identifier le type des colonnes.
    model_name : str
        Famille de modèle (voir :func:`build_estimator`).
    params : dict[str, Any]
        Hyperparamètres du classifieur.
    random_state : int
        Graine aléatoire.

    Returns
    -------
    Pipeline
        Pipeline non entraîné ``preprocessor -> model``.
    """
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(features)),
            ("model", build_estimator(model_name, params, random_state)),
        ]
    )


def create_cross_validation(n_splits: int, random_state: int) -> StratifiedKFold:
    """Crée le découpage de validation croisée stratifiée.

    Parameters
    ----------
    n_splits : int
        Nombre de plis.
    random_state : int
        Graine aléatoire du mélange des observations.

    Returns
    -------
    StratifiedKFold
        Découpage conservant la proportion de souscripteurs dans chaque pli.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def compute_oof_probabilities(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cross_validation: StratifiedKFold,
) -> np.ndarray:
    """Calcule les probabilités out-of-fold de souscription.

    Chaque observation est prédite par un modèle entraîné sans elle. Ces
    probabilités servent à choisir le seuil de décision sans utiliser le test.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline non entraîné.
    features : pd.DataFrame
        Variables explicatives d'entraînement.
    target : pd.Series
        Cible d'entraînement.
    cross_validation : StratifiedKFold
        Découpage de validation croisée.

    Returns
    -------
    np.ndarray
        Probabilité de souscription pour chaque observation.
    """
    probabilities = cross_val_predict(
        pipeline, features, target, cv=cross_validation, method="predict_proba"
    )[:, 1]
    logger.info("Probabilités OOF calculées pour %d observations", len(probabilities))
    return probabilities
