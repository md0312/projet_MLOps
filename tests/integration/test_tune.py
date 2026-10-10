import mlflow

from bank_marketing.application.mlflow_utils import get_tuning_trials, setup_mlflow
from bank_marketing.application.tune import run_tuning
from bank_marketing.infrastructure.config import load_config
from bank_marketing.settings import DEFAULT_CONFIG_PATH


def test_run_tuning_logs_one_child_run_per_trial(tmp_path):
    experiment_id = setup_mlflow(
        experiment_name="test_tuning",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifacts_dir=tmp_path / "mlruns",
    )

    best_config = run_tuning(load_config(DEFAULT_CONFIG_PATH), n_trials=2, results_dir=tmp_path)

    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    assert len(runs) == 3  # 1 run d'optimisation + 2 essais
    assert set(best_config["model"]["params"]) >= {"C", "l1_ratio", "class_weight"}
    assert (tmp_path / "tuned_logistic_regression.yaml").is_file()

    trials = get_tuning_trials("logistic_regression", experiment_name="test_tuning")
    assert trials["trial"].tolist() == [0, 1]
    assert {"C", "l1_ratio", "oof_pr_auc"} <= set(trials.columns)
