"""Entraînement du modèle final et choix de son seuil de décision.

Usage
-----
python -m bank_marketing.application.train --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from bank_marketing.application.preparation import prepare_train_test
from bank_marketing.domain.evaluation import compute_metrics, select_threshold
from bank_marketing.domain.modeling import (
    build_pipeline,
    compute_oof_probabilities,
    create_cross_validation,
)
from bank_marketing.infrastructure.artifacts import save_model
from bank_marketing.infrastructure.config import load_config
from bank_marketing.infrastructure.data import save_dataframe
from bank_marketing.settings import DEFAULT_CONFIG_PATH, MODELS_DIR, RESULTS_DIR
from bank_marketing.settings.logging_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = MODELS_DIR / "model.joblib"


def run_training(
    config: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, float]:
    """Entraîne le modèle configuré, choisit son seuil et l'enregistre.

    Le seuil est choisi sur les probabilités out-of-fold du jeu d'entraînement :
    le jeu de test n'intervient pas.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration du projet.
    model_path : Path, optional
        Fichier où enregistrer le modèle et son seuil.
    results_dir : Path, optional
        Dossier où enregistrer les métriques out-of-fold.

    Returns
    -------
    dict[str, float]
        Seuil retenu et métriques out-of-fold à ce seuil.
    """
    x_train, _, y_train, _ = prepare_train_test(config)
    model_config = config["model"]
    pipeline = build_pipeline(
        x_train, model_config["name"], model_config["params"], config["random_state"]
    )

    cross_validation = create_cross_validation(
        config["cross_validation"]["n_splits"], config["random_state"]
    )
    oof_probabilities = compute_oof_probabilities(pipeline, x_train, y_train, cross_validation)
    threshold = select_threshold(config["threshold"]["strategy"], y_train, oof_probabilities)
    oof_metrics = compute_metrics(y_train, oof_probabilities, threshold)
    logger.info("Seuil retenu : %.3f | F1 OOF : %.3f", threshold, oof_metrics["f1"])

    pipeline.fit(x_train, y_train)
    save_model(pipeline, threshold, model_path)

    results = {"threshold": threshold, **oof_metrics}
    save_dataframe(pd.DataFrame([results]), results_dir / "train_oof_metrics.csv")
    return results


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraîne le modèle final.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Fichier de configuration YAML."
    )
    args = parser.parse_args()

    configure_logging()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
