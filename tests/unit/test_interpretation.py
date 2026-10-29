import numpy as np
import pandas as pd
import pytest

from bank_marketing.domain.interpretation import (
    extract_logistic_coefficients,
    fit_inference_model,
    prepare_inference_features,
)
from bank_marketing.domain.modeling import build_pipeline


@pytest.fixture
def training_sample():
    """Jeu fictif où la souscription dépend fortement de l'âge."""
    rng = np.random.default_rng(0)
    age = rng.normal(45, 10, size=300)
    contact = rng.choice(["cellular", "telephone"], size=300)
    probability = 1 / (1 + np.exp(-(age - 45) / 5))
    target = pd.Series((rng.uniform(size=300) < probability).astype(int))
    return pd.DataFrame({"age": age, "contact": contact}), target


def test_extract_logistic_coefficients_names_and_sorts_variables(training_sample):
    features, target = training_sample
    pipeline = build_pipeline(features, "logistic_regression", {}, random_state=42)
    pipeline.fit(features, target)

    coefficients = extract_logistic_coefficients(pipeline)

    assert coefficients.index[0] == "age"
    assert set(coefficients.index) == {"age", "contact_cellular", "contact_telephone"}
    assert coefficients.loc["age", "odds_ratio"] > 1


def test_prepare_inference_features_removes_redundant_and_rare_values():
    features = pd.DataFrame(
        {
            "education": ["illiterate", "basic.4y", "high.school"],
            "loan": ["no", "yes", "no"],
            "previously_contacted": [0, 1, 0],
        }
    )

    prepared = prepare_inference_features(features)

    assert prepared.columns.tolist() == ["education"]
    assert "illiterate" not in prepared["education"].tolist()


def test_fit_inference_model_detects_significant_effect(training_sample):
    features, target = training_sample

    table = fit_inference_model(features, target)

    assert table.index[0] == "age"
    assert table.loc["age", "significatif_5pct"]
    assert table.loc["age", "or_ic95_bas"] > 1
