import pandas as pd
import pytest

from bank_marketing.domain.exploration import (
    compute_cramers_v,
    compute_subscription_rate,
    rank_categorical_associations,
    summarize_target,
    summarize_unknown_values,
)


@pytest.fixture
def raw_sample():
    """Extrait fictif : le contact par mobile est associé à la souscription."""
    return pd.DataFrame(
        {
            "contact": ["cellular", "cellular", "cellular", "telephone", "telephone", "telephone"],
            "default": ["no", "unknown", "no", "unknown", "no", "no"],
            "age": [30, 41, 35, 52, 47, 60],
            "y": ["yes", "yes", "no", "no", "no", "no"],
        }
    )


def test_summarize_target_counts_each_class(raw_sample):
    summary = summarize_target(raw_sample)

    assert summary.loc["no", "effectif"] == 4
    assert summary.loc["yes", "pourcentage"] == pytest.approx(33.33)


def test_summarize_unknown_values_only_lists_concerned_variables(raw_sample):
    summary = summarize_unknown_values(raw_sample)

    assert summary.index.tolist() == ["default"]
    assert summary.loc["default", "nombre_unknown"] == 2
    assert summary.loc["default", "taux_souscription_unknown_pct"] == 50.0


def test_compute_subscription_rate_by_category(raw_sample):
    rates = compute_subscription_rate(raw_sample, "contact")

    assert rates.loc["cellular", "souscriptions"] == 2
    assert rates.loc["cellular", "taux_souscription_pct"] == pytest.approx(66.67)
    assert rates.index[0] == "cellular"


def test_compute_cramers_v_is_one_for_identical_variables():
    values = pd.Series(["a", "b", "a", "b", "a", "b"] * 10)

    cramers_v, p_value = compute_cramers_v(values, values)

    assert cramers_v == pytest.approx(1.0, abs=0.05)
    assert p_value < 0.05


def test_rank_categorical_associations_sorts_by_strength(raw_sample):
    ranking = rank_categorical_associations(raw_sample, ["default", "contact"])

    assert ranking.index[0] == "contact"
    assert ranking["v_cramer"].is_monotonic_decreasing
