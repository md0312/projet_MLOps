"""Pipeline complet suivi par MLflow : entraînement, évaluation et enregistrement.

Chaque exécution crée un run MLflow contenant la configuration, les métriques
out-of-fold et de test, les fichiers de résultats et le modèle entraîné.

Usage
-----
python -m bank_marketing.application.run_pipeline --config configs/config.yaml
"""

import argparse
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from bank_marketing.application.evaluate import run_evaluation
from bank_marketing.application.mlflow_utils import (
    flatten_config,
    list_types_to_trust,
    prefix_metrics,
    setup_mlflow,
)
from bank_marketing.application.preparation import prepare_train_test
from bank_marketing.application.train import run_training
from bank_marketing.infrastructure.artifacts import load_model
from bank_marketing.infrastructure.config import load_config
from bank_marketing.settings import DEFAULT_CONFIG_PATH
from bank_marketing.settings.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def run_pipeline(config: dict[str, Any], config_path: Path) -> str:
    """Entraîne et évalue le modèle configuré dans un run MLflow.

    Le modèle et les résultats sont écrits dans un dossier temporaire puis
    enregistrés dans MLflow : les fichiers du modèle final (``models/`` et
    ``reports/results/``) ne sont pas modifiés.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration du projet.
    config_path : Path
        Fichier de configuration, enregistré comme artefact du run.

    Returns
    -------
    str
        Identifiant du run MLflow créé.
    """
    with (
        mlflow.start_run(run_name=config["model"]["name"]) as run,
        tempfile.TemporaryDirectory() as temporary_dir,
    ):
        output_dir = Path(temporary_dir)
        model_path = output_dir / "model.joblib"

        mlflow.set_tag("config_file", config_path.name)
        mlflow.log_params(flatten_config(config))

        train_results = run_training(config, model_path=model_path, results_dir=output_dir)
        test_results = run_evaluation(config, model_path=model_path, results_dir=output_dir)
        mlflow.log_metrics(prefix_metrics(train_results, "oof"))
        mlflow.log_metrics(prefix_metrics(test_results, "test"))

        mlflow.log_artifact(str(config_path), artifact_path="config")
        for csv_file in output_dir.glob("*.csv"):
            mlflow.log_artifact(str(csv_file), artifact_path="results")

        model, _ = load_model(model_path)
        _, x_test, _, _ = prepare_train_test(config)
        # La signature décrit les colonnes attendues par le modèle. MLflow avertit
        # que les colonnes entières ne peuvent pas contenir de valeurs manquantes :
        # les données ne contiennent aucune valeur manquante, il est donc ignoré.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
            signature = infer_signature(x_test, model.predict_proba(x_test)[:, 1])
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            skops_trusted_types=list_types_to_trust(model),
        )

        logger.info("Run MLflow terminé : %s", run.info.run_id)
        return run.info.run_id


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraîne et évalue un modèle suivi par MLflow.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Fichier de configuration YAML."
    )
    args = parser.parse_args()

    configure_logging()
    setup_mlflow()
    run_pipeline(load_config(args.config), args.config)


if __name__ == "__main__":
    main()
