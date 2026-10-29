"""Validation des données brutes par un schéma de contraintes.

Chaque colonne attendue est associée à une hypothèse : un intervalle de valeurs
pour les variables numériques, une liste de modalités pour les variables
catégorielles. La validation se fait à deux niveaux :

1. les lignes qui ne respectent pas le schéma sont isolées et écartées ;
2. si leur part dépasse un seuil de tolérance, la validation échoue, car le
   problème vient alors probablement de la source des données.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

UNBOUNDED = (float("-inf"), float("inf"))

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (17, 100),
    "duration": (0, float("inf")),
    "campaign": (1, float("inf")),
    # 999 signifie « jamais contacté auparavant » (voir le dictionnaire des données).
    "pdays": (0, 999),
    "previous": (0, float("inf")),
    # Indicateurs économiques : pas de borne métier, seule la présence d'une valeur
    # numérique est vérifiée.
    "emp.var.rate": UNBOUNDED,
    "cons.price.idx": UNBOUNDED,
    "cons.conf.idx": UNBOUNDED,
    "euribor3m": UNBOUNDED,
    "nr.employed": UNBOUNDED,
}

YES_NO_UNKNOWN = ("no", "unknown", "yes")

CATEGORY_VALUES: dict[str, tuple[str, ...]] = {
    "job": (
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ),
    "marital": ("divorced", "married", "single", "unknown"),
    "education": (
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    ),
    "default": YES_NO_UNKNOWN,
    "housing": YES_NO_UNKNOWN,
    "loan": YES_NO_UNKNOWN,
    "contact": ("cellular", "telephone"),
    "month": (
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ),
    "day_of_week": ("mon", "tue", "wed", "thu", "fri"),
    "poutcome": ("failure", "nonexistent", "success"),
    "y": ("no", "yes"),
}

EXPECTED_COLUMNS = [*NUMERIC_RANGES, *CATEGORY_VALUES]

DEFAULT_MAX_INVALID_SHARE = 0.05


class DataValidationError(ValueError):
    """Les données brutes ne respectent pas le schéma attendu."""


def check_structure(data: pd.DataFrame) -> None:
    """Vérifie la présence des colonnes attendues et le type des colonnes numériques.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes.

    Raises
    ------
    DataValidationError
        Si une colonne manque ou si une colonne numérique contient du texte.
    """
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise DataValidationError(f"Colonnes manquantes : {', '.join(missing_columns)}")

    non_numeric_columns = [
        column for column in NUMERIC_RANGES if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric_columns:
        raise DataValidationError(f"Colonnes non numériques : {', '.join(non_numeric_columns)}")


def find_invalid_rows(data: pd.DataFrame) -> pd.Series:
    """Repère les lignes qui ne respectent pas le schéma de contraintes.

    Une valeur manquante, hors de son intervalle ou de ses modalités rend la ligne
    invalide. Le nombre de lignes invalides est journalisé pour chaque colonne.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes, de structure déjà vérifiée.

    Returns
    -------
    pd.Series
        Booléen par ligne, vrai si la ligne est invalide.
    """
    invalid_rows = pd.Series(False, index=data.index)
    checks = {
        **{column: ~data[column].between(*bounds) for column, bounds in NUMERIC_RANGES.items()},
        **{column: ~data[column].isin(values) for column, values in CATEGORY_VALUES.items()},
    }
    for column, invalid_values in checks.items():
        if invalid_values.any():
            logger.warning("%s : %d valeur(s) hors du schéma", column, invalid_values.sum())
        invalid_rows |= invalid_values
    return invalid_rows


def validate_raw_data(
    data: pd.DataFrame, max_invalid_share: float = DEFAULT_MAX_INVALID_SHARE
) -> pd.DataFrame:
    """Valide les données brutes et écarte les lignes non conformes.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes.
    max_invalid_share : float, default=0.05
        Part maximale de lignes invalides tolérée.

    Returns
    -------
    pd.DataFrame
        Lignes conformes au schéma.

    Raises
    ------
    DataValidationError
        Si la structure est incorrecte ou si la part de lignes invalides dépasse
        le seuil de tolérance.
    """
    check_structure(data)
    invalid_rows = find_invalid_rows(data)
    invalid_share = invalid_rows.mean()
    if invalid_share > max_invalid_share:
        raise DataValidationError(
            f"{invalid_share:.1%} de lignes invalides (tolérance : {max_invalid_share:.0%})"
        )
    if invalid_rows.any():
        logger.warning("%d ligne(s) invalide(s) écartée(s)", invalid_rows.sum())
    logger.info("Données validées : %d lignes conformes au schéma", (~invalid_rows).sum())
    return data.loc[~invalid_rows]
