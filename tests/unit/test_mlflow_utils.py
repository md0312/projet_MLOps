from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from bank_marketing.application.mlflow_utils import (
    flatten_config,
    list_types_to_trust,
    prefix_metrics,
)


def test_flatten_config_joins_nested_keys_with_dots():
    config = {"random_state": 42, "model": {"name": "svm", "params": {"C": 1.0}}}

    flat = flatten_config(config)

    assert flat == {"random_state": 42, "model.name": "svm", "model.params.C": 1.0}


def test_prefix_metrics_renames_every_metric():
    metrics = {"f1": 0.485, "recall": 0.533}

    assert prefix_metrics(metrics, "test") == {"test_f1": 0.485, "test_recall": 0.533}


def test_list_types_to_trust_includes_calibration_types():
    model = CalibratedClassifierCV(SVC(), ensemble=False, cv=2)
    model.fit([[0.0], [1.0], [2.0], [3.0]], [0, 1, 0, 1])

    trusted_types = list_types_to_trust(model)

    assert "sklearn.calibration._CalibratedClassifier" in trusted_types
