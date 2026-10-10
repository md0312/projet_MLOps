from bank_marketing.domain.preprocessing import (
    build_modeling_dataset,
    create_preprocessor,
    split_train_test,
)
from bank_marketing.infrastructure.data import load_raw_data


def test_preprocessing_on_real_data_matches_exploration():
    features, target = build_modeling_dataset(load_raw_data())

    x_train, x_test, _, _ = split_train_test(features, target, test_size=0.2, random_state=42)
    preprocessor = create_preprocessor(x_train).fit(x_train)

    assert features.shape == (4119, 19)
    assert round(target.mean(), 4) == 0.1095
    assert (len(x_train), len(x_test)) == (3295, 824)
    assert preprocessor.transform(x_test).shape == (824, 62)
