"""Tests for plots generated only from saved result artifacts."""

import json
from pathlib import Path
import struct

import pandas as pd

from scripts import make_plots


def test_main_creates_plots_from_benchmark_comparison_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The plot command creates all five expected charts from CSV/JSON data."""
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    figures_dir = tmp_path / "docs" / "figures"
    monkeypatch.setattr(make_plots, "_RESULTS_DIR", results_dir)
    monkeypatch.setattr(make_plots, "_FIGURES_DIR", figures_dir)

    pd.DataFrame(
        {
            "n": [100, 100, 100, 1000, 1000, 1000],
            "index": ["numpy", "faiss-flat", "faiss-ivf"] * 2,
            "build_s": [0.01, 0.02, 0.03, 0.1, 0.2, 0.3],
            "search_ms_per_query": [0.1, 0.2, 0.05, 0.4, 0.6, 0.1],
            "memory_mb": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            "recall_vs_flat": [1.0, 1.0, 0.8, 1.0, 1.0, 0.7],
        }
    ).to_csv(results_dir / "benchmark_indexes.csv", index=False)
    pd.DataFrame(
        {
            "model": ["model-a", "model-b"],
            "recall@1": [0.2, 0.3],
            "recall@3": [0.4, 0.5],
            "recall@10": [0.6, 0.7],
            "mrr": [0.3, 0.4],
        }
    ).to_csv(results_dir / "model_comparison.csv", index=False)
    for index_type, recall, mrr in (
        ("numpy", 0.6, 0.4),
        ("faiss-flat", 0.6, 0.4),
        ("faiss-ivf", 0.5, 0.3),
    ):
        (results_dir / f"metrics_{index_type}.json").write_text(
            json.dumps({"metrics": {"recall@10": recall, "mrr": mrr}}),
            encoding="utf-8",
        )

    exit_code = make_plots.main()

    expected_names = {
        "benchmark_latency.png",
        "benchmark_build.png",
        "benchmark_recall_ivf.png",
        "model_comparison.png",
        "index_quality.png",
    }
    assert exit_code == 0
    assert {path.name for path in figures_dir.glob("*.png")} == expected_names
    for path in figures_dir.glob("*.png"):
        with path.open("rb") as image_file:
            header = image_file.read(24)
        width, height = struct.unpack(">II", header[16:24])
        assert width > 100
        assert height > 100
