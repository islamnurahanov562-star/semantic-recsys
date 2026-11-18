"""Slow end-to-end smoke test for the synthetic index benchmark."""

import pandas as pd
import pytest

from scripts.benchmark_indexes import main


@pytest.mark.slow
def test_main_writes_rows_for_all_indexes(tmp_path, monkeypatch) -> None:
    """A small synthetic run writes valid measurements for all index types."""
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "nested" / "benchmark_indexes.csv"

    exit_code = main(
        [
            "--sizes",
            "80",
            "--dim",
            "8",
            "--queries",
            "4",
            "--k",
            "3",
            "--repeats",
            "1",
            "--out",
            str(output_path),
        ]
    )

    results = pd.read_csv(output_path)
    assert exit_code == 0
    assert results["index"].tolist() == ["numpy", "faiss-flat", "faiss-ivf"]
    assert results["n"].tolist() == [80, 80, 80]
    assert results["recall_vs_flat"].between(0.0, 1.0).all()
    assert results.loc[results["index"] != "faiss-ivf", "recall_vs_flat"].eq(1.0).all()
    assert results["search_ms_per_query"].gt(0).all()
