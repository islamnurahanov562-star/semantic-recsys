# Data Card

## Source and License

- Dataset: [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows), by Shivam Bansal on Kaggle.
- Source file: `netflix_titles.csv`, downloaded locally to `data/raw/netflix_titles.csv`.
- License shown on the Kaggle dataset page: CC0: Public Domain. Checked 2026-11-15. The local download date was not recorded.
- The raw CSV is not committed or redistributed by this repository. CC0 is the dataset contributor's dedication and does not imply Netflix endorsement or transfer rights to Netflix marks or other third-party rights.

## Fields and Mapping

| Source field | Canonical field/use |
| --- | --- |
| `title` | `title`; included in embedding text |
| `description` | `description`; included in embedding text |
| `listed_in` | `category`; first genre retained in output, full value used while creating text |
| `release_year` | `year` |
| `type` | Included in embedding text |
| `rating` | Not used; it is a content rating, not a user rating |
| Other source fields | Not retained in the canonical item table |

## Processing

The implemented pipeline follows `configs/dataset.json` and the `TextPreprocessor` contract:

1. Missing text becomes empty text; HTML entities are unescaped, HTML tags removed, whitespace collapsed, and values stripped.
2. Empty or absent categories become `unknown`.
3. `text` concatenates `title`, `type`, `category`, and `description` with `. ` separators while the full genre field is available.
4. Rows with combined text shorter than 20 characters are removed.
5. Duplicate `(title, description)` pairs are removed case-insensitively, keeping the first occurrence.
6. Combined text is truncated to 1,000 characters at a word boundary.
7. Output category is reduced to the first comma-separated genre.
8. Rows are reindexed; `item_id` is assigned to the row position. `year` uses nullable `Int64` when present.

## Processed Statistics

Computed from the current `data/processed/items.csv`:

- Items: 8,804 rows and 6 columns.
- Distinct first-genre categories: 36.
- Top five first genres: Dramas 1,599; Comedies 1,209; Action & Adventure 859; Documentaries 829; International TV Shows 773.
- Description length: median 24 words; 95th percentile 28 words.
- Release year range: 1925–2021.

The current evaluation set in `data/processed/eval_queries.csv` contains 50 queries, averages 1.34 relevant IDs per query, and has 13 queries with multiple relevant IDs. Queries were written and reviewed manually; the builder checks IDs, duplicate queries, minimum length, and phrase leakage against each source description.

## Known Limitations

- There are no user-preference or user-rating labels, so the system performs semantic content retrieval rather than personalized recommendations.
- The evaluation set is small and relevance labels are human judgments; some labels may be incomplete, so measured Recall is not a definitive ceiling on model quality.
- The text and query set are English-oriented; multilingual performance is not established by the current results.
- Descriptions can be short and may not explicitly state every plot detail expressed by a query.
- Search quality varies with model, query phrasing, and approximate-index settings. The raw source file must be downloaded locally to reproduce preprocessing.
