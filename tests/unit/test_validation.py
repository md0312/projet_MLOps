import pandas as pd
import pytest

from bank_marketing.domain.validation import (
    EXPECTED_COLUMNS,
    DataValidationError,
    find_invalid_rows,
    validate_raw_data,
)


@pytest.fixture
def valid_data():
    """Vingt lignes fictives identiques, conformes au schéma."""
    row = {
        "age": 40,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 200,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
        "y": "no",
    }
    return pd.DataFrame([row] * 20)


def test_validate_raw_data_keeps_valid_data(valid_data):
    result = validate_raw_data(valid_data)

    pd.testing.assert_frame_equal(result, valid_data)


def test_validate_raw_data_rejects_missing_column(valid_data):
    with pytest.raises(DataValidationError, match="Colonnes manquantes : age"):
        validate_raw_data(valid_data.drop(columns="age"))


def test_validate_raw_data_rejects_text_in_numeric_column(valid_data):
    valid_data["age"] = "quarante"

    with pytest.raises(DataValidationError, match="Colonnes non numériques : age"):
        validate_raw_data(valid_data)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("age", -5),
        ("campaign", 0),
        ("pdays", 1500),
        ("job", "astronaut"),
        ("y", "maybe"),
        ("euribor3m", None),
    ],
)
def test_find_invalid_rows_detects_values_outside_schema(valid_data, column, invalid_value):
    valid_data.loc[3, column] = invalid_value

    invalid_rows = find_invalid_rows(valid_data)

    assert invalid_rows.tolist() == [index == 3 for index in range(20)]


def test_validate_raw_data_drops_invalid_rows_within_tolerance(valid_data):
    valid_data.loc[0, "age"] = 150

    result = validate_raw_data(valid_data, max_invalid_share=0.1)

    assert len(result) == 19
    assert 0 not in result.index


def test_validate_raw_data_fails_above_tolerance(valid_data):
    valid_data.loc[:4, "job"] = "astronaut"

    with pytest.raises(DataValidationError, match="25.0% de lignes invalides"):
        validate_raw_data(valid_data, max_invalid_share=0.1)


def test_expected_columns_match_raw_data_columns():
    assert len(EXPECTED_COLUMNS) == 21
