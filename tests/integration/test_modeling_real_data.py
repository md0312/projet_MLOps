import pytest

from bank_marketing.domain.evaluation import (
    build_confusion_table,
    compute_metrics,
    find_f1_threshold,
)
from bank_marketing.domain.modeling import (
    build_pipeline,
    compute_oof_probabilities,
    create_cross_validation,
)
from bank_marketing.domain.preprocessing import build_modeling_dataset, split_train_test
from bank_marketing.infrastructure.data import load_raw_data

# Réglage retenu lors de l'exploration (optimisation Optuna).
LOGISTIC_PARAMS = {
    "C": 0.043080746450545805,
    "l1_ratio": 0.04212694085665472,
    "solver": "saga",
    "class_weight": "balanced",
    "max_iter": 5000,
}


def test_final_logistic_regression_reproduces_exploration_results():
    features, target = build_modeling_dataset(load_raw_data())
    x_train, x_test, y_train, y_test = split_train_test(
        features, target, test_size=0.2, random_state=42
    )
    pipeline = build_pipeline(x_train, "logistic_regression", LOGISTIC_PARAMS, random_state=42)

    oof_probabilities = compute_oof_probabilities(
        pipeline, x_train, y_train, create_cross_validation(n_splits=5, random_state=42)
    )
    threshold = find_f1_threshold(y_train, oof_probabilities)
    test_probabilities = pipeline.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_probabilities, threshold)
    confusion = build_confusion_table(y_test, test_probabilities, threshold)

    assert threshold == pytest.approx(0.589, abs=1e-3)
    assert test_metrics["f1"] == pytest.approx(0.485, abs=1e-3)
    assert confusion.to_numpy().tolist() == [[674, 60], [42, 48]]
