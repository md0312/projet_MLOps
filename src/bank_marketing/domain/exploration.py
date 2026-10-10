"""Indicateurs de l'analyse exploratoire des données brutes.

Ces fonctions calculent les tableaux utilisés dans le notebook d'EDA. Elles
travaillent sur les données brutes, où la cible ``y`` vaut ``"yes"`` ou ``"no"``.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from bank_marketing.domain.preprocessing import POSITIVE_LABEL, TARGET_COLUMN

UNKNOWN_LABEL = "unknown"


def summarize_target(data: pd.DataFrame) -> pd.DataFrame:
    """Décrit la répartition de la cible.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes contenant la colonne cible ``y``.

    Returns
    -------
    pd.DataFrame
        Effectif et pourcentage de chaque modalité de la cible.
    """
    counts = data[TARGET_COLUMN].value_counts()
    return pd.DataFrame(
        {"effectif": counts, "pourcentage": (100 * counts / len(data)).round(2)}
    ).rename_axis("souscription")


def summarize_unknown_values(data: pd.DataFrame) -> pd.DataFrame:
    """Mesure la fréquence de la modalité ``unknown`` et son lien avec la cible.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes.

    Returns
    -------
    pd.DataFrame
        Pour chaque variable contenant ``unknown`` : nombre et part des
        ``unknown``, taux de souscription parmi eux et parmi les autres clients.
        Les variables sont triées par nombre décroissant de ``unknown``.
    """
    subscribed = data[TARGET_COLUMN].eq(POSITIVE_LABEL)
    rows = []
    for column in data.columns.drop(TARGET_COLUMN):
        is_unknown = data[column].eq(UNKNOWN_LABEL)
        if is_unknown.any():
            rows.append(
                {
                    "variable": column,
                    "nombre_unknown": int(is_unknown.sum()),
                    "part_unknown_pct": round(100 * is_unknown.mean(), 2),
                    "taux_souscription_unknown_pct": round(100 * subscribed[is_unknown].mean(), 2),
                    "taux_souscription_connu_pct": round(100 * subscribed[~is_unknown].mean(), 2),
                }
            )
    return pd.DataFrame(rows).sort_values("nombre_unknown", ascending=False).set_index("variable")


def compute_subscription_rate(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calcule l'effectif et le taux de souscription par modalité d'une variable.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes.
    column : str
        Variable catégorielle à analyser.

    Returns
    -------
    pd.DataFrame
        Effectif, nombre de souscriptions et taux de souscription (en %) par
        modalité, triés par taux décroissant.
    """
    return (
        data.assign(souscription=data[TARGET_COLUMN].eq(POSITIVE_LABEL))
        .groupby(column)
        .agg(effectif=("souscription", "size"), souscriptions=("souscription", "sum"))
        .assign(
            taux_souscription_pct=lambda table: (
                100 * table["souscriptions"] / table["effectif"]
            ).round(2)
        )
        .sort_values("taux_souscription_pct", ascending=False)
    )


def compute_cramers_v(first: pd.Series, second: pd.Series) -> tuple[float, float]:
    """Mesure l'association entre deux variables catégorielles.

    Parameters
    ----------
    first : pd.Series
        Première variable catégorielle.
    second : pd.Series
        Seconde variable catégorielle.

    Returns
    -------
    tuple[float, float]
        Le V de Cramér, entre 0 (aucune association) et 1 (association
        parfaite), et la p-value du test du chi-deux d'indépendance.
    """
    contingency_table = pd.crosstab(first, second)
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    n_observations = contingency_table.to_numpy().sum()
    min_dimension = min(contingency_table.shape) - 1
    return float(np.sqrt(chi2 / (n_observations * min_dimension))), float(p_value)


def rank_categorical_associations(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Classe des variables catégorielles selon leur association avec la cible.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes.
    columns : list[str]
        Variables catégorielles à évaluer.

    Returns
    -------
    pd.DataFrame
        V de Cramér et p-value du chi-deux pour chaque variable, triés par
        association décroissante.
    """
    rows = []
    for column in columns:
        cramers_v, p_value = compute_cramers_v(data[column], data[TARGET_COLUMN])
        rows.append({"variable": column, "v_cramer": round(cramers_v, 3), "p_value": p_value})
    return pd.DataFrame(rows).sort_values("v_cramer", ascending=False).set_index("variable")
