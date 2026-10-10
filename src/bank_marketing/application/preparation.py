"""Préparation des jeux d'entraînement et de test à partir de la configuration."""

from typing import Any

import pandas as pd

from bank_marketing.domain.preprocessing import (
    build_modeling_dataset,
    select_features,
    split_train_test,
)
from bank_marketing.infrastructure.data import load_raw_data


def prepare_train_test(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Charge les données brutes et produit les jeux d'entraînement et de test.

    Le découpage dépend uniquement de la configuration (graine et proportion du
    test) : l'entraînement et l'évaluation retrouvent donc exactement les mêmes
    jeux.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration du projet (voir ``configs/config.yaml``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``X_train``, ``X_test``, ``y_train`` et ``y_test``.
    """
    features, target = build_modeling_dataset(load_raw_data())
    features = select_features(features, config["features"]["feature_set"])
    return split_train_test(
        features,
        target,
        test_size=config["data"]["test_size"],
        random_state=config["random_state"],
    )
