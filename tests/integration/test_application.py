import pytest

from bank_marketing.application.evaluate import run_evaluation
from bank_marketing.application.train import run_training
from bank_marketing.infrastructure.config import load_config
from bank_marketing.settings import DEFAULT_CONFIG_PATH


def test_training_then_evaluation_reproduce_exploration_results(tmp_path):
    config = load_config(DEFAULT_CONFIG_PATH)
    model_path = tmp_path / "model.joblib"

    train_results = run_training(config, model_path=model_path, results_dir=tmp_path)
    test_results = run_evaluation(config, model_path=model_path, results_dir=tmp_path)

    assert train_results["threshold"] == pytest.approx(0.589, abs=1e-3)
    assert train_results["f1"] == pytest.approx(0.487, abs=1e-3)
    assert test_results["f1"] == pytest.approx(0.485, abs=1e-3)
    assert (tmp_path / "test_metrics.csv").is_file()
    assert (tmp_path / "test_confusion_matrix.csv").is_file()
