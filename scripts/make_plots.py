"""Create publication-ready plots from saved benchmark and evaluation files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("results")
_FIGURES_DIR = Path("docs/figures")
_INDEXES = ("numpy", "faiss-flat", "faiss-ivf")
_METRICS = ("recall@1", "recall@3", "recall@10", "mrr")


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Apply consistent spacing, save a PNG, and close its figure.

    Args:
        figure: Matplotlib figure to save.
        output_path: Destination PNG path.
    """
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_benchmark(benchmark: pd.DataFrame) -> list[Path]:
    """Create synthetic latency, build-time, and IVF-recall plots.

    Args:
        benchmark: Benchmark rows with size, index, build, search, memory, recall.

    Returns:
        Generated benchmark chart paths.
    """
    generated: list[Path] = []
    required = {
        "n",
        "index",
        "build_s",
        "search_ms_per_query",
        "memory_mb",
        "recall_vs_flat",
    }
    missing = required.difference(benchmark.columns)
    if missing:
        logger.warning("Skipping benchmark plots; missing columns: %s", sorted(missing))
        return generated

    fig, axis = plt.subplots(figsize=(8, 5))
    for index_type, rows in benchmark.groupby("index", sort=True):
        rows = rows.sort_values("n")
        axis.plot(
            rows["n"],
            rows["search_ms_per_query"],
            marker="o",
            label=str(index_type),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Index Search Latency")
    axis.set_xlabel("Number of synthetic vectors (n)")
    axis.set_ylabel("Median search time per query (ms)")
    axis.legend(title="Index")
    fig.suptitle("Synthetic vectors")
    output_path = _FIGURES_DIR / "benchmark_latency.png"
    _save_figure(fig, output_path)
    generated.append(output_path)

    fig, axis = plt.subplots(figsize=(8, 5))
    for index_type, rows in benchmark.groupby("index", sort=True):
        rows = rows.sort_values("n")
        axis.plot(rows["n"], rows["build_s"], marker="o", label=str(index_type))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Index Build Time")
    axis.set_xlabel("Number of synthetic vectors (n)")
    axis.set_ylabel("Build time (s)")
    axis.legend(title="Index")
    fig.suptitle("Synthetic vectors")
    output_path = _FIGURES_DIR / "benchmark_build.png"
    _save_figure(fig, output_path)
    generated.append(output_path)

    ivf_rows = benchmark.loc[benchmark["index"].eq("faiss-ivf")].sort_values("n")
    if not ivf_rows.empty:
        fig, axis = plt.subplots(figsize=(8, 5))
        sizes = ivf_rows["n"].astype(int).to_numpy()
        positions = range(len(sizes))
        axis.bar(positions, ivf_rows["recall_vs_flat"].to_numpy())
        axis.set_xticks(positions, labels=[f"{size:,}" for size in sizes])
        axis.set_ylim(0, 1.05)
        axis.set_title("FAISS IVF Recall Relative to Flat")
        axis.set_xlabel("Number of synthetic vectors (n)")
        axis.set_ylabel("Recall@k vs. FAISS Flat")
        output_path = _FIGURES_DIR / "benchmark_recall_ivf.png"
        _save_figure(fig, output_path)
        generated.append(output_path)
    return generated


def _plot_model_comparison(comparison: pd.DataFrame) -> Path | None:
    """Create grouped retrieval-quality bars for each evaluated model.

    Args:
        comparison: Model-comparison result rows.

    Returns:
        The chart path, or None if required measurements are absent.
    """
    required = {"model", *_METRICS}
    missing = required.difference(comparison.columns)
    if missing:
        logger.warning(
            "Skipping model comparison plot; missing columns: %s", sorted(missing)
        )
        return None

    plot_data = comparison.set_index("model").loc[:, list(_METRICS)].copy()
    plot_data.index = [str(model).split("/")[-1] for model in plot_data.index]
    ax = plot_data.plot.bar(figsize=(10, 5), width=0.82)
    ax.set_title("Embedding Model Retrieval Quality")
    ax.set_xlabel("Embedding model")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="Metric")
    output_path = _FIGURES_DIR / "model_comparison.png"
    _save_figure(ax.figure, output_path)
    return output_path


def _plot_index_quality() -> Path | None:
    """Plot Recall@10 and MRR from saved per-index metric JSON files.

    Returns:
        The chart path, or None if no index metric artifact is available.
    """
    rows: list[dict[str, str | float]] = []
    for index_type in _INDEXES:
        metrics_path = _RESULTS_DIR / f"metrics_{index_type}.json"
        if not metrics_path.is_file():
            logger.warning("Skipping missing index metrics: %s", metrics_path)
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            rows.append(
                {
                    "index": index_type,
                    "recall@10": float(metrics["recall@10"]),
                    "mrr": float(metrics["mrr"]),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("Skipping invalid metrics %s: %s", metrics_path, error)

    if not rows:
        logger.warning("Skipping index quality plot; no usable metric files found.")
        return None

    ax = pd.DataFrame(rows).set_index("index").plot.bar(figsize=(8, 5), width=0.8)
    ax.set_title("Retrieval Quality by Index")
    ax.set_xlabel("Vector index")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric")
    output_path = _FIGURES_DIR / "index_quality.png"
    _save_figure(ax.figure, output_path)
    return output_path


def main() -> int:
    """Read available results, create plots, and print their paths.

    Returns:
        Zero if at least one plot was produced, otherwise one.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    benchmark_path = _RESULTS_DIR / "benchmark_indexes.csv"
    if benchmark_path.is_file():
        generated.extend(_plot_benchmark(pd.read_csv(benchmark_path)))
    else:
        logger.warning("Skipping benchmark plots; %s does not exist.", benchmark_path)

    comparison_path = _RESULTS_DIR / "model_comparison.csv"
    if comparison_path.is_file():
        comparison_plot = _plot_model_comparison(pd.read_csv(comparison_path))
        if comparison_plot is not None:
            generated.append(comparison_plot)
    else:
        logger.warning("Skipping model plot; %s does not exist.", comparison_path)

    index_quality_plot = _plot_index_quality()
    if index_quality_plot is not None:
        generated.append(index_quality_plot)

    print("Created plots:")
    for path in generated:
        print(path)
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
