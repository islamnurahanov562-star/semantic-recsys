# Semantic Recommendation Engine: Technical Report Plan

## Abstract

TODO: Write a concise summary of the problem, system, evaluation method, principal measured results, and limitations after experiments are complete.

<!-- TODO(owner: R4): Write the abstract after the method and results are finalized; source: completed report sections and measured artifacts under results/. -->

## 1 Introduction

Keyword search matches literal terms and may fail to retrieve titles that express the same idea using different words. This project investigates local semantic search over Netflix titles using text embeddings and vector similarity. The objective and scope should be finalized against the implemented system before publication.

### Contributions

- TODO: Describe the implemented data-processing and embedding pipeline.
- TODO: Describe the implemented and compared vector indexes.
- TODO: Report measured retrieval quality and runtime results only after evaluation.
- TODO: State each team member's verified contribution.

<!-- TODO(owner: R4): Verify the contribution statements with the team; source: merged pull requests and team review. -->

<!-- TODO(owner: R4): Finalize the problem statement, objective, and verified contribution list; source: CONTRACTS.md, implemented modules, and team review. -->

## 2 Related Work

Sentence-BERT adapts BERT with siamese and triplet network structures to produce sentence embeddings that can be compared efficiently. This work motivates representing title metadata and descriptions as vectors for semantic retrieval (Reimers & Gurevych, 2019).

FAISS provides methods and implementations for efficient similarity search over dense vectors. Its exact and inverted-file index approaches motivate the index variants and benchmarking considered in this project (Johnson, Douze, & Jégou, 2017).

<!-- TODO(owner: R3): Verify the bibliographic details and connect each cited work to the implemented design; source: the two references listed below. -->

## 3 Method

<!-- TODO(owner: R4): Describe the implemented end-to-end method and keep it consistent with the code; source: CONTRACTS.md and the modules listed below. -->

### 3.1 Data

Describe the Netflix Movies and TV Shows dataset, its provenance, the fields used, and the preprocessing contract. The canonical item fields are `title`, `description`, `category`, optional `year`, and embedding `text`; document missing-value handling and duplicate removal from the actual preprocessing behavior. Dataset URL, license, and data-card details: TODO.

<!-- TODO(owner: R2): Document verified dataset provenance, selected fields, preprocessing decisions, and limitations; source: configs/dataset.json, docs/DATA_CARD.md (TODO), and data/processed/items.csv (TODO). -->

### 3.2 Embeddings

Describe the selected SentenceTransformers model, the text supplied to it, vector dimensionality, normalization, and embedding persistence. Explain that L2-normalized vectors permit inner-product ranking as cosine similarity. Model choice, dimensionality, and measured properties: TODO.

<!-- TODO(owner: R3): Record the actual model, embedding configuration, and measured properties; source: src/encoder.py and embedding metadata under data/processed/ (TODO). -->

### 3.3 Indexes

Describe the NumPy exact-search baseline, FAISS Flat exact search, and FAISS IVF approximate search. Explain their shared `BaseIndex` interface, index construction and persistence, and the trade-offs measured in the experiments. Implementation details and measured trade-offs: TODO.

<!-- TODO(owner: R3): Describe the implemented index variants and their behavior without claiming unmeasured speedups; source: src/index.py and benchmark artifacts under results/ (TODO). -->

### 3.4 System Architecture and OOP Design

Explain the responsibilities of `DataLoader`, `TextPreprocessor`, `EmbeddingEncoder`, `BaseIndex`, `NumpyIndex`, `FaissIndex`, `RecommendationEngine`, and `Evaluator`. Describe inheritance through `BaseIndex`, polymorphic index use by the engine, and encapsulation at the class boundaries. Distinguish implemented modules from planned modules in the final report.

<!-- TODO(owner: R1): Verify the architecture description against the integrated implementation and explain the OOP boundaries; source: CONTRACTS.md and src/. -->

### 3.5 Evaluation Protocol

- **Recall@k:** for each query, `|relevant ∩ top-k| / |relevant|`; report the mean across queries.
- **MRR:** mean reciprocal rank of the first relevant item within the contract's top-ten cutoff; use zero when no relevant item is found.
- **Latency:** measure each `engine.recommend()` call with `perf_counter`, report elapsed time in milliseconds, and average across evaluation queries.

The evaluation set contains 50 manually reviewed queries. Relevance labels were selected by searching item descriptions with pandas `str.contains` and inspecting matching descriptions; neither the embedding model nor a vector index was used to choose labels. The run configuration, measured results, and per-query outputs are saved in [`eval_summary.json`](../results/eval_summary.json) and [`retrieval_quality.csv`](../results/retrieval_quality.csv).

## 4 Experiments

<!-- TODO(owner: R3): Describe only experiments that were actually run and retain their settings; source: experiment scripts and artifacts under results/ (TODO). -->

### 4.1 Setup

Hardware and operating environment: TODO. Python and library versions: TODO. Record the model, index parameters, dataset version, and any benchmark controls used in the actual runs.

<!-- TODO(owner: R3): Record reproducible hardware, software versions, model, and index settings; source: environment metadata and benchmark configuration (TODO). -->

### 4.2 Index Benchmark

Compare NumPy, FAISS Flat, and FAISS IVF using measured search latency and memory. Dataset sizes, repetitions, warm-up policy, and hardware must be reported from the actual benchmark setup; results: TODO.

<!-- TODO(owner: R3): Populate this subsection from completed index benchmarks and explain the benchmark protocol; source: results/TODO_index_benchmark.csv. -->

### 4.3 Model Comparison

Compare the embedding models actually evaluated, including their configuration and measured retrieval results. Model names, settings, and results: TODO.

<!-- TODO(owner: R3): Describe only models that were run and provide the measured comparison; source: results/TODO_model_comparison.csv. -->

### 4.4 Retrieval Quality

The measured comparison uses 8,804 processed items and `sentence-transformers/all-MiniLM-L6-v2`. One untimed warm-up query per index precedes latency measurement with `perf_counter`; reported CPU latency is machine- and load-dependent. Results for NumPy, FAISS Flat, and FAISS IVF on the 50 reviewed queries are shown in Section 5 and are reproducible with `python -m scripts.evaluate_engine`.

### 4.5 Error Analysis

Inspect queries with no relevant result or low recall, group observed failure patterns, and discuss likely causes supported by examples. Error categories and examples: TODO.

<!-- TODO(owner: R4): Analyze actual per-query outputs without inventing examples or causes; source: Evaluator per-query output and results/TODO_error_analysis.csv. -->

## 5 Results

Only the retrieval evaluation below has been run so far; index benchmarks and model comparisons remain TODO until measured.

### Index Benchmark Results

<!-- TODO(owner: R3): Populate the table from measured index benchmark artifacts; source: results/TODO_index_benchmark.csv. -->

| Index | Search latency | Memory | Retrieval quality | Artifact |
| --- | --- | --- | --- | --- |
| NumPy | TODO | TODO | TODO | [TODO: add measured benchmark file](../results/TODO_index_benchmark.csv) |
| FAISS Flat | TODO | TODO | TODO | [TODO: add measured benchmark file](../results/TODO_index_benchmark.csv) |
| FAISS IVF | TODO | TODO | TODO | [TODO: add measured benchmark file](../results/TODO_index_benchmark.csv) |

### Embedding Model Results

<!-- TODO(owner: R3): Populate the table from actual model comparison runs; source: results/TODO_model_comparison.csv. -->

| Model | Retrieval quality | Artifact |
| --- | --- | --- |
| TODO | TODO | [TODO: add measured model comparison file](../results/TODO_model_comparison.csv) |

### Retrieval Evaluation Results

<!-- TODO(owner: R4): Populate the table from the reviewed evaluation set and measured outputs; source: results/TODO_retrieval_quality.csv. -->

| Model / index | Queries | Recall@1 / @3 / @10 | HitRate@1 / @3 / @10 | MRR | Average latency (ms) | Artifacts |
| --- | ---: | --- | --- | ---: | ---: | --- |
| all-MiniLM-L6-v2 / NumPy | 50 | 0.2100 / 0.3200 / 0.4217 | 0.2400 / 0.3600 / 0.4800 | 0.3116 | 197.9339 | [summary](../results/eval_summary.json), [per-query](../results/retrieval_quality.csv) |
| all-MiniLM-L6-v2 / FAISS Flat | 50 | 0.2100 / 0.3200 / 0.4217 | 0.2400 / 0.3600 / 0.4800 | 0.3116 | 66.4140 | [summary](../results/eval_summary.json), [per-query](../results/retrieval_quality.csv) |
| all-MiniLM-L6-v2 / FAISS IVF | 50 | 0.1900 / 0.2700 / 0.3817 | 0.2200 / 0.3000 / 0.4400 | 0.2821 | 68.0107 | [summary](../results/eval_summary.json), [per-query](../results/retrieval_quality.csv) |

<!-- TODO(owner: R3): Add benchmark and model-comparison measurements only after those experiments run. -->

## 6 Limitations

Discuss dataset coverage and licensing, the quality and size of human-reviewed relevance labels, model and language limitations, approximate-search trade-offs, hardware dependence, and any unimplemented components. Confirm each limitation against project evidence; discussion: TODO.

<!-- TODO(owner: R4): Describe limitations supported by the actual dataset, implementation, and experiments; source: docs/DATA_CARD.md (TODO), evaluation artifacts, and code review. -->

## 7 Conclusion and Future Work

Summarize what was implemented and measured, relate conclusions to the project objectives, and list evidence-based next steps. Conclusions, measured findings, and future-work priorities: TODO.

<!-- TODO(owner: R4): Write conclusions only after results are finalized and list realistic next steps; source: completed report sections and results/. -->

## References

- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
- Johnson, J., Douze, M., & Jégou, H. (2017). *Billion-scale similarity search with GPUs*.
- Netflix Movies and TV Shows dataset, Kaggle. Dataset page and license: TODO.

<!-- TODO(owner: R4): Verify citation metadata and add the confirmed dataset citation; source: the cited publications and verified Kaggle dataset page (TODO). -->