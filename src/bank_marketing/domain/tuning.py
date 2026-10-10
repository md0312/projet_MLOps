"""Espaces de recherche des hyperparamètres pour l'optimisation avec Optuna."""

from typing import Any

import optuna

TUNABLE_MODELS = ("logistic_regression", "random_forest")


def suggest_hyperparameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Propose un jeu d'hyperparamètres pour un essai Optuna.

    Seuls les paramètres explorés sont tirés au hasard ; les paramètres fixes
    (pondération des classes, solveur, etc.) sont toujours les mêmes.

    Parameters
    ----------
    trial : optuna.Trial
        Essai Optuna en cours.
    model_name : {"logistic_regression", "random_forest"}
        Famille de modèle à optimiser.

    Returns
    -------
    dict[str, Any]
        Hyperparamètres complets du classifieur.

    Raises
    ------
    ValueError
        Si le modèle n'a pas d'espace de recherche défini.
    """
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "solver": "saga",
            "class_weight": "balanced",
            "max_iter": 5000,
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
            "class_weight": "balanced",
            "n_jobs": -1,
        }
    raise ValueError(
        f"Pas d'espace de recherche pour {model_name!r} (attendu : {', '.join(TUNABLE_MODELS)})"
    )
