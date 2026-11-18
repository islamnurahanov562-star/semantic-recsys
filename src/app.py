"""Streamlit interface for local semantic search over catalog items."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from src.factory import build_engine

if TYPE_CHECKING:
    from src.engine import RecommendationEngine

logger = logging.getLogger(__name__)
_ITEMS_PATH = Path("data/processed/items.csv")
_EVAL_PATH = Path("data/processed/eval_queries.csv")
_INDEX_OPTIONS = ("faiss-flat", "numpy", "faiss-ivf")


@st.cache_data
def _load_items() -> pd.DataFrame:
    """Load processed item metadata for category and description display.

    Returns:
        Processed item catalog.
    """
    return pd.read_csv(_ITEMS_PATH)


@st.cache_resource(show_spinner=False)
def load_engine(index_type: str) -> RecommendationEngine:
    """Load and cache one recommendation engine per index type.

    Args:
        index_type: One of ``numpy``, ``faiss-flat``, or ``faiss-ivf``.

    Returns:
        A ready-to-use recommendation engine.
    """
    engine = build_engine(index_type=index_type)
    engine.recommend("a story about family and friendship", k=1)
    return engine


@st.cache_data
def _load_example_queries() -> tuple[str, ...]:
    """Load up to four reviewed example queries from the evaluation set.

    Returns:
        Non-empty example query strings.
    """
    if not _EVAL_PATH.is_file():
        return ()
    queries = pd.read_csv(_EVAL_PATH, usecols=["query"], dtype={"query": "string"})[
        "query"
    ]
    return tuple(queries.dropna().str.strip().loc[lambda values: values.ne("")].head(4))


def _set_query(query: str) -> None:
    """Set the query widget value from an example button callback.

    Args:
        query: Example query text.
    """
    st.session_state["query_input"] = query


def render_sidebar(items: pd.DataFrame) -> tuple[str, int, str | None]:
    """Render index, result-count, and category controls.

    Args:
        items: Catalog used to populate category options.

    Returns:
        Selected index type, requested result count, and optional category.
    """
    with st.sidebar:
        st.header("Search settings")
        index_type = st.selectbox(
            "Vector index",
            options=_INDEX_OPTIONS,
            index=0,
            format_func=lambda value: {
                "faiss-flat": "FAISS Flat",
                "numpy": "NumPy exact",
                "faiss-ivf": "FAISS IVF",
            }[value],
        )
        result_count = st.slider("Results", min_value=1, max_value=10, value=3)
        categories = sorted(items["category"].dropna().astype(str).unique().tolist())
        selected_category = st.selectbox("Category", ["All", *categories])
        st.divider()
        st.subheader("About")
        st.caption(
            "Find titles by the meaning of their descriptions, "
            "not only by exact keywords."
        )
    return (
        index_type,
        result_count,
        None if selected_category == "All" else selected_category,
    )


def render_examples() -> None:
    """Render four buttons that fill the query field with reviewed examples."""
    examples = _load_example_queries()
    if not examples:
        return
    st.caption("Try a reviewed query")
    columns = st.columns(2)
    for index, query in enumerate(examples):
        columns[index % 2].button(
            query,
            key=f"example_query_{index}",
            help=query,
            use_container_width=True,
            on_click=_set_query,
            args=(query,),
        )


def render_results(recommendations: pd.DataFrame) -> None:
    """Render recommendations as compact, inspectable result panels.

    Args:
        recommendations: Ranked result rows returned by the engine.
    """
    if recommendations.empty:
        st.info("No matching titles were found for these settings.")
        return

    for result in recommendations.itertuples(index=False):
        with st.container(border=True):
            title_column, score_column = st.columns([5, 1])
            title_column.markdown(f"**{int(result.rank)}. {result.title}**")
            title_column.caption(str(result.category))
            score = float(result.score)
            score_column.metric("Similarity", f"{score:.3f}")
            st.progress(min(max(score, 0.0), 1.0))
            description = "" if pd.isna(result.description) else str(result.description)
            with st.expander("Read more"):
                st.write(description[:300])


def main() -> None:
    """Render the search app and handle submitted queries."""
    st.set_page_config(page_title="Semantic Recommendation Engine", layout="wide")
    st.session_state.setdefault("query_input", "")
    st.title("Semantic Recommendation Engine")

    try:
        items = _load_items()
    except (OSError, ValueError) as error:
        logger.exception("Could not load catalog data.")
        st.error(f"Could not load catalog data: {error}")
        return

    index_type, result_count, category = render_sidebar(items)
    st.text_input(
        "Describe the title or story you are looking for",
        key="query_input",
        placeholder="A heartfelt story about a family starting over...",
    )
    submitted = st.button("Search", type="primary", use_container_width=True)
    render_examples()

    if not submitted:
        return
    query = str(st.session_state["query_input"]).strip()
    if not query:
        st.warning("Enter a search query first.")
        return

    try:
        with st.spinner("Loading model and index..."):
            engine = load_engine(index_type)
        start_time = time.perf_counter()
        recommendations = engine.recommend(query, k=result_count, category=category)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        logger.exception("Recommendation search failed.")
        st.error(f"Search could not be completed: {error}")
        return

    render_results(recommendations)
    st.caption(f"Search time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
