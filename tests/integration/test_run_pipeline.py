import mlflow
import pytest

from bank_marketing.application.mlflow_utils import setup_mlflow
from bank_marketing.application.run_pipeline import run_pipeline
from bank_marketing.infrastructure.config import load_config
from bank_marketing.settings import DEFAULT_CONFIG_PATH, MLFLOW_REGISTERED_MODEL_NAME


def test_run_pipeline_logs_and_registers_model(tmp_path):
    setup_mlflow(
        experiment_name="test",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifacts_dir=tmp_path / "mlruns",
    )

    run_id = run_pipeline(load_config(DEFAULT_CONFIG_PATH), DEFAULT_CONFIG_PATH)

    run = mlflow.get_run(run_id)
    assert run.data.params["model.name"] == "logistic_regression"
    assert run.data.metrics["oof_threshold"] == pytest.approx(0.589, abs=1e-3)
    assert run.data.metrics["test_f1"] == pytest.approx(0.485, abs=1e-3)
    assert mlflow.sklearn.load_model(f"runs:/{run_id}/model") is not None

    client = mlflow.MlflowClient()
    [version] = client.search_model_versions(f"name='{MLFLOW_REGISTERED_MODEL_NAME}'")
    assert version.run_id == run_id
    assert float(version.tags["threshold"]) == pytest.approx(0.589, abs=1e-3)
    assert version.tags["model_name"] == "logistic_regression"
