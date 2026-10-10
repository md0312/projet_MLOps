import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from bank_marketing.domain.modeling import (
    build_estimator,
    build_pipeline,
    compute_oof_probabilities,
    create_cross_validation,
)


@pytest.fixture
def training_sample():
    """Petit jeu d'entraînement fictif, avec une variable numérique et une catégorielle."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "age": rng.integers(20, 70, size=40),
            "contact": rng.choice(["cellular", "telephone"], size=40),
        }
    )
    target = pd.Series([0, 1] * 20)
    return features, target


@pytest.mark.parametrize(
    ("model_name", "params", "expected_type"),
    [
        ("logistic_regression", {"C": 0.5}, LogisticRegression),
        ("random_forest", {"n_estimators": 10}, RandomForestClassifier),
        ("svm", {"C": 1.0}, CalibratedClassifierCV),
    ],
)
def test_build_estimator_returns_expected_model(model_name, params, expected_type):
    estimator = build_estimator(model_name, params, random_state=42)

    assert isinstance(estimator, expected_type)


def test_build_estimator_applies_hyperparameters():
    estimator = build_estimator("logistic_regression", {"C": 0.043}, random_state=7)

    assert estimator.C == 0.043
    assert estimator.random_state == 7


def test_build_estimator_rejects_unknown_model():
    with pytest.raises(ValueError, match="inconnu"):
        build_estimator("xgboost", {}, random_state=42)


def test_build_pipeline_chains_preprocessor_and_model(training_sample):
    features, _ = training_sample

    pipeline = build_pipeline(features, "logistic_regression", {}, random_state=42)

    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_compute_oof_probabilities_returns_one_probability_per_row(training_sample):
    features, target = training_sample
    pipeline = build_pipeline(features, "logistic_regression", {}, random_state=42)

    probabilities = compute_oof_probabilities(
        pipeline, features, target, create_cross_validation(n_splits=4, random_state=42)
    )

    assert probabilities.shape == (40,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
