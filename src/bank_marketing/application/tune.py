"""Optimisation des hyperparamètres avec Optuna, suivie dans MLflow.

Chaque essai Optuna est enregistré comme un run MLflow « enfant » du run
d'optimisation. Le critère optimisé est le PR-AUC des probabilités out-of-fold
du jeu d'entraînement : le jeu de test n'est jamais utilisé.

Usage
-----
python -m bank_marketing.application.tune --config configs/config.yaml --n-trials 40
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import mlflow
import optuna
from sklearn.metrics import average_precision_score

from bank_marketing.application.mlflow_utils import flatten_config, setup_mlflow
from bank_marketing.application.preparation import prepare_train_test
from bank_marketing.application.train import compute_config_oof_probabilities
from bank_marketing.domain.tuning import suggest_hyperparameters
from bank_marketing.infrastructure.config import load_config, save_config
from bank_marketing.settings import DEFAULT_CONFIG_PATH, RESULTS_DIR
from bank_marketing.settings.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def run_tuning(
    config: dict[str, Any], n_trials: int, results_dir: Path = RESULTS_DIR
) -> dict[str, Any]:
    """Cherche les meilleurs hyperparamètres du modèle configuré.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration de base : famille de modèle, jeu de variables, plis.
    n_trials : int
        Nombre d'essais Optuna.
    results_dir : Path, optional
        Dossier où enregistrer la configuration optimisée.

    Returns
    -------
    dict[str, Any]
        Configuration complète avec les meilleurs hyperparamètres trouvés.
    """
    model_name = config["model"]["name"]
    x_train, _, y_train, _ = prepare_train_test(config)

    def objective(trial: optuna.Trial) -> float:
        """Évalue un jeu d'hyperparamètres par le PR-AUC out-of-fold."""
        trial_config = copy.deepcopy(config)
        trial_config["model"]["params"] = suggest_hyperparameters(trial, model_name)

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            probabilities = compute_config_oof_probabilities(trial_config, x_train, y_train)
            pr_auc = float(average_precision_score(y_train, probabilities))
            mlflow.log_params(trial.params)
            mlflow.log_metric("oof_pr_auc", pr_auc)
        return pr_auc

    with mlflow.start_run(run_name=f"tuning_{model_name}"):
        mlflow.set_tag("type", "tuning")
        mlflow.log_params({**flatten_config(config), "n_trials": n_trials})

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config["random_state"]),
        )
        study.optimize(objective, n_trials=n_trials)

        best_config = copy.deepcopy(config)
        best_config["model"]["params"] = suggest_hyperparameters(
            optuna.trial.FixedTrial(study.best_params), model_name
        )
        mlflow.log_params({f"best.{name}": value for name, value in study.best_params.items()})
        mlflow.log_metric("best_oof_pr_auc", study.best_value)

        output_path = results_dir / f"tuned_{model_name}.yaml"
        save_config(best_config, output_path)
        mlflow.log_artifact(str(output_path), artifact_path="config")

    logger.info("Meilleur PR-AUC OOF : %.3f | paramètres : %s", study.best_value, study.best_params)
    return best_config


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Optimise les hyperparamètres avec Optuna.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Fichier de configuration YAML."
    )
    parser.add_argument("--n-trials", type=int, default=40, help="Nombre d'essais Optuna.")
    args = parser.parse_args()

    configure_logging()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setup_mlflow()
    run_tuning(load_config(args.config), args.n_trials)


if __name__ == "__main__":
    main()
