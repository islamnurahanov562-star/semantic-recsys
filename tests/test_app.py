"""Smoke tests for the Streamlit search interface."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"


def test_app_renders_search_controls_without_loading_model() -> None:
    """The initial page shows search controls and reviewed examples."""
    app = AppTest.from_file(_APP_PATH, default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Semantic Recommendation Engine"
    assert app.button[0].label == "Search"
    assert len(app.button) == 5
    assert app.selectbox[0].value == "faiss-flat"
    assert app.slider[0].value == 3


def test_app_warns_when_search_is_submitted_empty() -> None:
    """Submitting an empty query displays a warning without loading a model."""
    app = AppTest.from_file(_APP_PATH, default_timeout=10).run()

    app.button[0].click().run()

    assert not app.exception
    assert any(
        "Enter a search query first." in warning.value for warning in app.warning
    )
