"""Évaluation unique du modèle entraîné sur le jeu de test.

Usage
-----
python -m bank_marketing.application.evaluate --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from bank_marketing.application.preparation import prepare_train_test
from bank_marketing.application.train import DEFAULT_MODEL_PATH
from bank_marketing.domain.evaluation import build_confusion_table, compute_metrics
from bank_marketing.infrastructure.artifacts import load_model
from bank_marketing.infrastructure.config import load_config
from bank_marketing.infrastructure.data import save_dataframe
from bank_marketing.settings import DEFAULT_CONFIG_PATH, RESULTS_DIR
from bank_marketing.settings.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def run_evaluation(
    config: dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, float]:
    """Évalue le modèle enregistré sur le jeu de test, avec son seuil.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration utilisée lors de l'entraînement (même découpage).
    model_path : Path, optional
        Fichier du modèle créé par l'entraînement.
    results_dir : Path, optional
        Dossier où enregistrer les métriques et la matrice de confusion.

    Returns
    -------
    dict[str, float]
        Seuil utilisé et métriques sur le jeu de test.
    """
    _, x_test, _, y_test = prepare_train_test(config)
    model, threshold = load_model(model_path)

    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_probabilities, threshold)
    confusion = build_confusion_table(y_test, test_probabilities, threshold)
    logger.info("F1 test : %.3f | rappel : %.3f", test_metrics["f1"], test_metrics["recall"])

    results = {"threshold": threshold, **test_metrics}
    # Arrondi à 6 décimales : évite des écarts infimes entre systèmes d'exploitation.
    save_dataframe(pd.DataFrame([results]).round(6), results_dir / "test_metrics.csv")
    save_dataframe(confusion.reset_index(names="reel"), results_dir / "test_confusion_matrix.csv")
    return results


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Évalue le modèle sur le jeu de test.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Fichier de configuration YAML."
    )
    args = parser.parse_args()

    configure_logging()
    run_evaluation(load_config(args.config))


if __name__ == "__main__":
    main()
