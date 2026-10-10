import pandas as pd
import pytest

from bank_marketing.domain.preprocessing import (
    INTERPRETABLE_FEATURES,
    add_previous_contact_flag,
    build_modeling_dataset,
    create_preprocessor,
    select_features,
    split_train_test,
)


@pytest.fixture
def raw_sample():
    """Petit extrait fictif ayant la même structure que les données brutes."""
    return pd.DataFrame(
        {
            "age": [30, 45, 52, 28],
            "job": ["admin.", "unknown", "retired", "student"],
            "duration": [120, 300, 45, 600],
            "pdays": [999, 6, 999, 3],
            "euribor3m": [4.8, 1.3, 4.9, 0.7],
            "y": ["no", "yes", "no", "yes"],
        }
    )


def test_add_previous_contact_flag_detects_previous_contacts(raw_sample):
    result = add_previous_contact_flag(raw_sample)

    assert result["previously_contacted"].tolist() == [0, 1, 0, 1]


def test_add_previous_contact_flag_does_not_modify_input(raw_sample):
    add_previous_contact_flag(raw_sample)

    assert "previously_contacted" not in raw_sample.columns


def test_build_modeling_dataset_excludes_leakage_and_target(raw_sample):
    features, _ = build_modeling_dataset(raw_sample)

    assert {"y", "duration", "pdays"}.isdisjoint(features.columns)
    assert "previously_contacted" in features.columns


def test_build_modeling_dataset_encodes_target_as_binary(raw_sample):
    _, target = build_modeling_dataset(raw_sample)

    assert target.tolist() == [0, 1, 0, 1]


def test_select_features_interpretable_keeps_expected_columns():
    features = pd.DataFrame(columns=[*INTERPRETABLE_FEATURES, "nr.employed", "day_of_week"])

    result = select_features(features, "interpretable")

    assert result.columns.tolist() == INTERPRETABLE_FEATURES


def test_select_features_rejects_unknown_set():
    with pytest.raises(ValueError, match="inconnu"):
        select_features(pd.DataFrame(), "unknown_set")


def test_split_train_test_is_stratified():
    features = pd.DataFrame({"x": range(100)})
    target = pd.Series([1] * 20 + [0] * 80)

    _, _, y_train, y_test = split_train_test(features, target, test_size=0.2, random_state=42)

    assert len(y_test) == 20
    assert y_train.mean() == y_test.mean() == 0.2


def test_create_preprocessor_scales_numeric_and_encodes_categories(raw_sample):
    features, _ = build_modeling_dataset(raw_sample)

    transformed = create_preprocessor(features).fit_transform(features)

    # 3 numériques (age, euribor3m, previously_contacted) + 4 modalités de job
    assert transformed.shape == (4, 7)


def test_create_preprocessor_ignores_unseen_category(raw_sample):
    features, _ = build_modeling_dataset(raw_sample)
    preprocessor = create_preprocessor(features).fit(features)
    new_client = features.head(1).assign(job="astronaut")

    transformed = preprocessor.transform(new_client)

    assert transformed.shape == (1, 7)
