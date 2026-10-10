import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

from bank_marketing.infrastructure.artifacts import load_model, save_model
from bank_marketing.infrastructure.config import load_config
from bank_marketing.infrastructure.data import load_raw_data, save_dataframe
from bank_marketing.infrastructure.figures import save_figure


def test_load_raw_data_reads_semicolon_separated_file(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text('"age";"y"\n30;"no"\n45;"yes"\n', encoding="utf-8")

    data = load_raw_data(csv_path)

    assert data.shape == (2, 2)
    assert list(data.columns) == ["age", "y"]


def test_load_raw_data_default_file_has_expected_shape():
    data = load_raw_data()

    assert data.shape == (4119, 21)
    assert "y" in data.columns


def test_load_raw_data_raises_when_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "missing.csv")


def test_save_dataframe_creates_parent_folder(tmp_path):
    output_path = tmp_path / "results" / "metrics.csv"
    metrics = pd.DataFrame({"metric": ["f1"], "value": [0.485]})

    save_dataframe(metrics, output_path)

    pd.testing.assert_frame_equal(pd.read_csv(output_path), metrics)


def test_save_and_load_model_keep_model_and_threshold(tmp_path):
    model = LogisticRegression().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    model_path = tmp_path / "models" / "model.joblib"

    save_model(model, threshold=0.589, path=model_path)
    loaded_model, loaded_threshold = load_model(model_path)

    assert loaded_threshold == 0.589
    assert (loaded_model.predict([[0.0], [3.0]]) == [0, 1]).all()


def test_load_model_raises_when_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_load_config_returns_dictionary(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("random_state: 42\nmodel:\n  name: logistic_regression\n", "utf-8")

    config = load_config(config_path)

    assert config == {"random_state": 42, "model": {"name": "logistic_regression"}}


def test_save_figure_writes_png_file(tmp_path):
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])

    path = save_figure(figure, "test_figure", directory=tmp_path)

    assert path == tmp_path / "test_figure.png"
    assert path.stat().st_size > 0
