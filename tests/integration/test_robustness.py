import pandas as pd
import pytest

from bank_marketing.application.robustness import run_seed_robustness
from bank_marketing.infrastructure.config import load_config
from bank_marketing.settings import DEFAULT_CONFIG_PATH


def test_seed_robustness_reruns_pipeline_for_each_seed(tmp_path):
    summary = run_seed_robustness(
        load_config(DEFAULT_CONFIG_PATH), seeds=[42, 7], results_dir=tmp_path
    )

    seed_results = pd.read_csv(tmp_path / "seed_robustness.csv").set_index("seed")
    # La graine du projet retrouve exactement les résultats du modèle final.
    assert seed_results.loc[42, "threshold"] == pytest.approx(0.589, abs=1e-3)
    assert seed_results.loc[42, "test_f1"] == pytest.approx(0.485, abs=1e-3)
    assert seed_results.loc[7, "test_f1"] != pytest.approx(0.485, abs=1e-3)
    assert summary.loc["test_f1", "ecart_type"] > 0
    assert (tmp_path / "seed_robustness_summary.csv").is_file()
