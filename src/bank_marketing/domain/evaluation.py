"""Métriques d'évaluation et choix du seuil de décision.

La cible étant déséquilibrée (environ 11 % de souscripteurs), l'accuracy n'est
pas utilisée. Les modèles sont comparés avec des métriques indépendantes du
seuil (ROC-AUC, PR-AUC) puis au seuil optimisé propre à chacun.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_THRESHOLD = 0.5


def find_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Renvoie le seuil de probabilité qui maximise le F1-score.

    Parameters
    ----------
    y_true : pd.Series
        Cible binaire observée.
    probabilities : np.ndarray
        Probabilités prédites de la classe positive.

    Returns
    -------
    float
        Seuil maximisant le F1-score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    # Le dernier couple (précision, rappel) n'a pas de seuil associé.
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1_values)])


def find_youden_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Renvoie le seuil qui maximise l'indice de Youden (sensibilité + spécificité - 1).

    Parameters
    ----------
    y_true : pd.Series
        Cible binaire observée.
    probabilities : np.ndarray
        Probabilités prédites de la classe positive.

    Returns
    -------
    float
        Seuil maximisant l'indice de Youden.
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probabilities)
    return float(thresholds[np.argmax(true_positive_rate - false_positive_rate)])


def compute_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Calcule les métriques de classification à un seuil donné.

    Parameters
    ----------
    y_true : pd.Series
        Cible binaire observée.
    probabilities : np.ndarray
        Probabilités prédites de la classe positive.
    threshold : float
        Seuil de décision : une souscription est prédite si la probabilité
        est supérieure ou égale à ce seuil.

    Returns
    -------
    dict[str, float]
        ROC-AUC et PR-AUC (indépendants du seuil), précision, rappel et F1.
    """
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
    }


def summarize_thresholds(
    model_name: str, y_true: pd.Series, probabilities: np.ndarray
) -> pd.DataFrame:
    """Compare les métriques aux seuils 0,5, F1 optimal et Youden.

    Parameters
    ----------
    model_name : str
        Nom du modèle, reporté dans le tableau.
    y_true : pd.Series
        Cible binaire observée.
    probabilities : np.ndarray
        Probabilités prédites (idéalement out-of-fold).

    Returns
    -------
    pd.DataFrame
        Une ligne par seuil, avec sa valeur et les métriques associées.
    """
    thresholds = {
        "0,50": DEFAULT_THRESHOLD,
        "F1 optimal": find_f1_threshold(y_true, probabilities),
        "Youden": find_youden_threshold(y_true, probabilities),
    }
    rows = [
        {
            "modele": model_name,
            "seuil": threshold_name,
            "valeur_seuil": threshold,
            **compute_metrics(y_true, probabilities, threshold),
        }
        for threshold_name, threshold in thresholds.items()
    ]
    return pd.DataFrame(rows).round(3)


def build_confusion_table(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Construit la matrice de confusion lisible à un seuil donné.

    Parameters
    ----------
    y_true : pd.Series
        Cible binaire observée.
    probabilities : np.ndarray
        Probabilités prédites de la classe positive.
    threshold : float
        Seuil de décision.

    Returns
    -------
    pd.DataFrame
        Matrice de confusion avec des libellés explicites.
    """
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        confusion_matrix(y_true, predictions, labels=[0, 1]),
        index=["Réel : non", "Réel : oui"],
        columns=["Prédit : non", "Prédit : oui"],
    )
