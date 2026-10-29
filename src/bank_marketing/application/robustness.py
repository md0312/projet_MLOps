"""Robustesse des résultats à la graine aléatoire.

Le pipeline complet (découpage, validation croisée, choix du seuil, entraînement
et évaluation) est relancé avec plusieurs graines. La dispersion des métriques
indique si les résultats obtenus avec la graine du projet sont représentatifs ou
s'ils dépendent du hasard du découpage. Les hyperparamètres restent ceux de la
configuration.

Usage
-----
python -m bank_marketing.application.robustness --config configs/config.yaml --n-seeds 10
"""

import argparse
import copy
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from bank_marketing.application.evaluate import run_evaluation
from bank_marketing.application.train import run_training
from bank_marketing.domain.evaluation import summarize_seed_results
from bank_marketing.infrastructure.config import load_config
from bank_marketing.infrastructure.data import save_dataframe
from bank_marketing.settings import DEFAULT_CONFIG_PATH, RESULTS_DIR
from bank_marketing.settings.logging_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_N_SEEDS = 10


def run_one_seed(config: dict[str, Any], seed: int) -> dict[str, float]:
    """Entraîne et évalue le modèle configuré avec une graine donnée.

    Le modèle et ses résultats sont écrits dans un dossier temporaire : les
    fichiers du modèle final ne sont pas modifiés.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration du projet.
    seed : int
        Graine aléatoire remplaçant celle de la configuration.

    Returns
    -------
    dict[str, float]
        Graine, seuil, métriques out-of-fold (préfixe ``oof_``) et de test
        (préfixe ``test_``).
    """
    seed_config = copy.deepcopy(config)
    seed_config["random_state"] = seed
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        model_path = output_dir / "model.joblib"
        train_results = run_training(seed_config, model_path=model_path, results_dir=output_dir)
        test_results = run_evaluation(seed_config, model_path=model_path, results_dir=output_dir)
    threshold = train_results.pop("threshold")
    test_results.pop("threshold")
    return {
        "seed": seed,
        "threshold": threshold,
        **{f"oof_{name}": value for name, value in train_results.items()},
        **{f"test_{name}": value for name, value in test_results.items()},
    }


def run_seed_robustness(
    config: dict[str, Any],
    seeds: list[int],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Relance le pipeline pour chaque graine et résume la dispersion des métriques.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration du projet.
    seeds : list[int]
        Graines aléatoires à tester.
    results_dir : Path, optional
        Dossier où enregistrer les résultats par graine et leur résumé.

    Returns
    -------
    pd.DataFrame
        Résumé des métriques : moyenne, écart-type, minimum et maximum.
    """
    seed_results = pd.DataFrame([run_one_seed(config, seed) for seed in seeds]).round(6)
    summary = summarize_seed_results(seed_results.drop(columns="seed"))
    summary_table = summary.reset_index(names="metrique")
    save_dataframe(seed_results, results_dir / "seed_robustness.csv")
    save_dataframe(summary_table, results_dir / "seed_robustness_summary.csv")
    logger.info(
        "F1 test sur %d graines : %.3f ± %.3f",
        len(seeds),
        summary.loc["test_f1", "moyenne"],
        summary.loc["test_f1", "ecart_type"],
    )
    return summary


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Mesure la sensibilité des résultats à la graine.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Fichier de configuration YAML."
    )
    parser.add_argument(
        "--n-seeds", type=int, default=DEFAULT_N_SEEDS, help="Nombre de graines à tester."
    )
    args = parser.parse_args()

    configure_logging()
    run_seed_robustness(load_config(args.config), seeds=list(range(args.n_seeds)))


if __name__ == "__main__":
    main()
