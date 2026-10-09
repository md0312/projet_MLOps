import numpy as np
import pandas as pd
import pytest

from bank_marketing.domain.evaluation import (
    build_confusion_table,
    compute_metrics,
    find_f1_threshold,
    find_youden_threshold,
    summarize_thresholds,
)


@pytest.fixture
def predictions():
    """Cible et probabilités fictives, parfaitement séparables au seuil 0,6."""
    y_true = pd.Series([0, 0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.55, 0.6, 0.9])
    return y_true, probabilities


def test_find_f1_threshold_separates_classes(predictions):
    y_true, probabilities = predictions

    assert find_f1_threshold(y_true, probabilities) == 0.6


def test_find_youden_threshold_separates_classes(predictions):
    y_true, probabilities = predictions

    assert find_youden_threshold(y_true, probabilities) == 0.6


def test_compute_metrics_at_default_threshold(predictions):
    y_true, probabilities = predictions

    metrics = compute_metrics(y_true, probabilities, threshold=0.5)

    # Au seuil 0,5, le client à 0,55 devient un faux positif.
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_without_positive_prediction_does_not_fail(predictions):
    y_true, probabilities = predictions

    metrics = compute_metrics(y_true, probabilities, threshold=0.99)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_summarize_thresholds_has_one_row_per_threshold(predictions):
    y_true, probabilities = predictions

    summary = summarize_thresholds("modele_test", y_true, probabilities)

    assert summary["seuil"].tolist() == ["0,50", "F1 optimal", "Youden"]
    assert summary.loc[1, "f1"] == 1.0


def test_build_confusion_table_counts_each_case(predictions):
    y_true, probabilities = predictions

    table = build_confusion_table(y_true, probabilities, threshold=0.5)

    assert table.loc["Réel : non", "Prédit : oui"] == 1
    assert table.loc["Réel : oui", "Prédit : oui"] == 2
    assert table.to_numpy().sum() == 6
