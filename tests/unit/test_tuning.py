import optuna
import pytest

from bank_marketing.domain.tuning import suggest_hyperparameters


def test_suggest_hyperparameters_for_logistic_regression_keeps_fixed_parameters():
    trial = optuna.trial.FixedTrial({"C": 0.05, "l1_ratio": 0.1})

    params = suggest_hyperparameters(trial, "logistic_regression")

    assert params["C"] == 0.05
    assert params["class_weight"] == "balanced"
    assert params["solver"] == "saga"


def test_suggest_hyperparameters_for_random_forest():
    trial = optuna.trial.FixedTrial(
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.5}
    )

    params = suggest_hyperparameters(trial, "random_forest")

    assert params["max_depth"] == 4
    assert params["class_weight"] == "balanced"


def test_suggest_hyperparameters_rejects_model_without_search_space():
    with pytest.raises(ValueError, match="espace de recherche"):
        suggest_hyperparameters(optuna.trial.FixedTrial({}), "svm")
