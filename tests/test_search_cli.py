"""Tests for the one-shot and interactive search CLI."""

import pandas as pd
import pytest

from scripts import search_cli


class FakeEngine:
    """Return deterministic search results and record requested options."""

    def __init__(self) -> None:
        """Initialize a call log for search assertions."""
        self.calls: list[tuple[str, int, str | None]] = []

    def recommend(
        self,
        query: str,
        k: int = 3,
        category: str | None = None,
    ) -> pd.DataFrame:
        """Return fixture results, or reject a designated invalid query."""
        self.calls.append((query, k, category))
        if query == "invalid":
            raise ValueError("query must not be empty")
        return pd.DataFrame(
            {
                "rank": [1],
                "item_id": [10],
                "title": ["A title"],
                "category": ["Drama"],
                "score": [0.87654],
                "description": ["A" * 150],
            }
        )


def test_one_shot_search_formats_results_and_forwards_options(
    monkeypatch,
    capsys,
) -> None:
    """One-shot mode prints the requested category, k, and formatted scores."""
    engine = FakeEngine()
    builds: list[tuple[str, str]] = []

    def fake_build_engine(*, model_name: str, index_type: str) -> FakeEngine:
        builds.append((model_name, index_type))
        return engine

    monkeypatch.setattr(search_cli, "build_engine", fake_build_engine)

    exit_code = search_cli.main(
        [
            "--query",
            "story in space",
            "--k",
            "5",
            "--index",
            "numpy",
            "--category",
            "Drama",
            "--model",
            "local-model",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert builds == [("local-model", "numpy")]
    assert engine.calls == [("story in space", 5, "Drama")]
    assert "0.877" in output
    assert "A title" in output
    assert "Search time:" in output
    assert "A" * 121 not in output


@pytest.mark.parametrize("terminator", ["", "exit"])
def test_interactive_mode_reuses_one_engine_until_exit(
    monkeypatch,
    capsys,
    terminator: str,
) -> None:
    """Interactive mode searches repeatedly using one built engine."""
    engine = FakeEngine()
    builds: list[bool] = []
    queries = iter(["first query", "second query", terminator])

    monkeypatch.setattr(
        search_cli, "build_engine", lambda **kwargs: builds.append(True) or engine
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(queries))

    exit_code = search_cli.main(["--index", "faiss-flat", "--k", "2"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len(builds) == 1
    assert engine.calls == [
        ("first query", 2, None),
        ("second query", 2, None),
    ]
    assert output.count("Search time:") == 2


def test_invalid_query_prints_friendly_message(
    monkeypatch,
    capsys,
) -> None:
    """A ValueError from recommend is printed without a traceback."""
    engine = FakeEngine()
    monkeypatch.setattr(search_cli, "build_engine", lambda **kwargs: engine)

    exit_code = search_cli.main(["--query", "invalid"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Invalid query: query must not be empty" in output
    assert "Traceback" not in output


def test_engine_build_value_error_prints_friendly_message(
    monkeypatch,
    capsys,
) -> None:
    """Factory configuration errors are reported without a traceback."""

    def fail_build_engine(**kwargs: str) -> FakeEngine:
        raise ValueError("invalid index")

    monkeypatch.setattr(search_cli, "build_engine", fail_build_engine)

    exit_code = search_cli.main(["--query", "story"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Could not start search: invalid index" in output
    assert "Traceback" not in output
