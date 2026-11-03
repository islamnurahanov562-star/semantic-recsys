# %%
# 1. Load the raw CSV and inspect its structure.
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "data/raw/netflix_titles.csv"
TITLE_COL = "title"
DESC_COL = "description"
CATEGORY_COL = "listed_in"
YEAR_COL = "release_year"
TYPE_COL = "type"

data_file = Path(DATA_PATH)
try:
    df = pd.read_csv(data_file, encoding="utf-8")
except UnicodeDecodeError:
    warnings.warn(
        "UTF-8 decoding failed; retrying the CSV with latin-1 encoding.",
        UnicodeWarning,
        stacklevel=2,
    )
    df = pd.read_csv(data_file, encoding="latin-1")

print(f"Shape: {df.shape}")
print("\nColumn dtypes:")
print(df.dtypes.to_string())
memory_bytes = int(df.memory_usage(index=True, deep=True).sum())
print(f"\nMemory usage: {memory_bytes:,} bytes ({memory_bytes / (1024 ** 2):.2f} MiB)")

# %%
# 2. Summarize missing values by column.
row_count = max(len(df), 1)
missing_table = pd.DataFrame(
    {
        "column": df.columns,
        "missing_count": df.isna().sum().to_numpy(),
        "missing_percent": (df.isna().sum().to_numpy() / row_count) * 100,
    }
).sort_values("missing_count", ascending=False)
print(
    missing_table.to_string(
        index=False, formatters={"missing_percent": "{:.2f}%".format}
    )
)

# %%
# 3. Count fully duplicated rows and repeated identifiers/titles.
full_duplicate_rows = int(df.duplicated().sum())
normalized_titles = df[TITLE_COL].astype("string").str.strip().str.casefold()
repeated_title_rows = int(
    normalized_titles[normalized_titles.notna()].duplicated(keep=False).sum()
)
repeated_show_ids = int(df["show_id"].dropna().duplicated().sum())

print(f"Fully duplicate rows (excluding the first occurrence): {full_duplicate_rows}")
print(f"Rows with a repeated lowercase title: {repeated_title_rows}")
print(f"Repeated show_id values (excluding the first occurrence): {repeated_show_ids}")

# %%
# 4. Describe description lengths in characters and words.
description_text = df[DESC_COL].fillna("").astype("string")
description_char_lengths = description_text.str.len().fillna(0).astype("int64")
description_word_lengths = (
    description_text.str.split().str.len().fillna(0).astype("int64")
)
quantiles = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
statistic_labels = ["min", "5%", "25%", "median", "75%", "95%", "max"]
length_statistics = pd.DataFrame(
    {
        "characters": np.quantile(description_char_lengths, quantiles),
        "words": np.quantile(description_word_lengths, quantiles),
    },
    index=statistic_labels,
)
print(length_statistics.to_string())
short_non_missing_descriptions = int(
    (description_char_lengths.lt(20) & df[DESC_COL].notna()).sum()
)
print(
    "\nNon-missing descriptions shorter than 20 characters: "
    f"{short_non_missing_descriptions}"
)

# %%
# 5. Show the distribution of Netflix content types.
type_distribution = df[TYPE_COL].value_counts(dropna=False)
print(type_distribution.to_string())

# %%
# 6. Compare first-genre and all-genre frequencies.
genre_lists = df[CATEGORY_COL].fillna("").astype("string").str.split(",")
first_genres = genre_lists.map(
    lambda genres: genres[0].strip() if genres and genres[0].strip() else pd.NA
)
all_genres = genre_lists.explode().str.strip()
all_genres = all_genres[all_genres.notna() & all_genres.ne("")]
genre_counts_per_record = genre_lists.map(
    lambda genres: sum(bool(genre.strip()) for genre in genres)
)

print("Top 15 first genres:")
print(first_genres.value_counts().head(15).to_string())
print("\nTop 15 genres across all listed positions:")
print(all_genres.value_counts().head(15).to_string())
print(f"\nMean genres per record: {genre_counts_per_record.mean():.2f}")

# %%
# 7. Inspect release-year range and display its distribution.
release_years = pd.to_numeric(df[YEAR_COL], errors="coerce").dropna()
print(f"Minimum release year: {release_years.min()}")
print(f"Maximum release year: {release_years.max()}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(release_years, bins=30, edgecolor="white")
ax.set_title("Distribution of Release Years")
ax.set_xlabel("Release year")
ax.set_ylabel("Number of titles")
fig.tight_layout()
plt.show()
plt.close(fig)

# %%
# 8. Review a reproducible sample of raw records.
sample_rows = df.sample(n=min(10, len(df)), random_state=42)
sample_view = sample_rows[[TITLE_COL, TYPE_COL, CATEGORY_COL, DESC_COL]].copy()
sample_view[DESC_COL] = (
    sample_view[DESC_COL].fillna("").astype("string").str.slice(0, 200)
)
print(sample_view.to_string(index=False))

# %%
# 9. Save the four requested EDA figures.
figures_dir = Path("docs/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(max(8, len(missing_table) * 0.7), 5))
ax.bar(missing_table["column"].astype(str), missing_table["missing_percent"])
ax.set_title("Missing Values by Column")
ax.set_xlabel("Column")
ax.set_ylabel("Missing values (%)")
ax.tick_params(axis="x", labelrotation=45)
fig.tight_layout()
fig.savefig(figures_dir / "eda_missing.png", dpi=150, bbox_inches="tight")
plt.close(fig)

word_length_cap = float(description_word_lengths.quantile(0.99))
capped_word_lengths = description_word_lengths.clip(upper=word_length_cap)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(capped_word_lengths, bins=30, edgecolor="white")
ax.set_title("Description Length in Words (Capped at 99th Percentile)")
ax.set_xlabel("Words per description")
ax.set_ylabel("Number of titles")
fig.tight_layout()
fig.savefig(figures_dir / "eda_desc_length.png", dpi=150, bbox_inches="tight")
plt.close(fig)

top_genres = all_genres.value_counts().head(15).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_genres.index.astype(str), top_genres.values)
ax.set_title("Top 15 Genres Across All Listed Positions")
ax.set_xlabel("Number of titles")
ax.set_ylabel("Genre")
fig.tight_layout()
fig.savefig(figures_dir / "eda_top_genres.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(release_years, bins=30, edgecolor="white")
ax.set_title("Distribution of Release Years")
ax.set_xlabel("Release year")
ax.set_ylabel("Number of titles")
fig.tight_layout()
fig.savefig(figures_dir / "eda_years.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved EDA figures to: {figures_dir}")

# %%
# 10. Add evidence-based observations after reviewing the outputs.
print("FINDINGS")
print("1. TODO: ...")
print("2. TODO: ...")
print("3. TODO: ...")
print("4. TODO: ...")
print("5. TODO: ...")
