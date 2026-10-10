import pytest

from bank_marketing.application.evaluate import run_evaluation
from bank_marketing.application.preparation import prepare_train_test
from bank_marketing.application.train import compute_config_oof_probabilities, run_training
from bank_marketing.infrastructure.config import load_config
from bank_marketing.settings import CONFIGS_DIR, DEFAULT_CONFIG_PATH


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


@pytest.mark.parametrize("config_name", ["config.yaml", "random_forest.yaml", "svm.yaml"])
def test_every_configuration_produces_valid_oof_probabilities(config_name):
    config = load_config(CONFIGS_DIR / config_name)
    x_train, _, y_train, _ = prepare_train_test(config)

    probabilities = compute_config_oof_probabilities(config, x_train, y_train)

    assert probabilities.shape == (len(y_train),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
