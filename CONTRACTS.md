# CONTRACTS (единый источник правды; менять может только Team Lead)

## Общие правила
- Python 3.10+, type hints везде, docstrings в стиле Google, PEP 8, black (длина строки 88).
- Код в src/ не использует print(), только модуль logging. В scripts/ print разрешён.
- Seed всегда 42. Пути только через pathlib.Path. Абсолютных путей в коде нет.
- Совместимость с pandas 3.x и numpy 2.x: без inplace=True, без chained assignment, типы приводим явно.
- Скрипты запускаются из корня репозитория: python -m scripts.<имя>
- Тесты: pytest -m "not slow". Тесты, которые скачивают модели или идут дольше 5 секунд, помечаются @pytest.mark.slow.
- Код и docstrings на английском, пояснения для людей на русском.

## Датасет
- Источник: Kaggle "Netflix Movies and TV Shows", файл data/raw/netflix_titles.csv (в git не хранится).
- Соответствие колонок: title -> title, description -> description, listed_in -> category, release_year -> year, type используется только для текстового поля. Колонка rating НЕ используется (это возрастной рейтинг).
- category в items.csv = первый жанр из listed_in; полный список жанров остаётся в text.
- text = title, type, listed_in, description, склеенные через ". ".

## Контракт данных
- data/processed/items.csv, колонки: item_id (int, уникальный, равен позиции строки 0..n-1), title (str), description (str), category (str, "unknown" если пусто), year (Int64, необязательно), text (str, поле для эмбеддингов).
- data/processed/eval_queries.csv, колонки: query (str), relevant_ids (str, item_id через "|", например "12|45|78").
- Эмбеддинги: np.ndarray float32 формы (n, d), L2-нормированные, строка i соответствует item_id i. Файл data/processed/embeddings_<slug>.npy, где slug = model_name.split("/")[-1], рядом embeddings_<slug>.meta.json.
- Отсутствующий результат поиска: индекс -1 и score -inf.

## Контракты классов (модуль: класс(сигнатура))
- src/data_loader.py: DataLoader(path: str | Path, required_columns: Sequence[str], column_map: dict[str, str] | None = None). Методы: load() -> pd.DataFrame (переименовывает колонки по column_map, затем валидирует); validate(df) -> None (ValueError при проблемах).
- src/preprocessor.py: TextPreprocessor(text_columns: Sequence[str], max_chars: int = 1000, min_chars: int = 20). Методы: transform(df) -> pd.DataFrame (чистит, убирает дубликаты, сбрасывает item_id, добавляет "text", оставляет первый жанр в category); атрибут report_ (dict со счётчиками удалённых строк); staticmethod save(df, path) -> None.
- src/encoder.py: EmbeddingEncoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64, device: str | None = None). Свойство dim -> int. Методы: encode(texts: Sequence[str], show_progress: bool = False) -> np.ndarray (float32, нормированный); encode_query(query: str) -> np.ndarray формы (1, d); staticmethods save_embeddings(emb, path), load_embeddings(path) -> np.ndarray.
- src/index.py: абстрактный BaseIndex(ABC): build(vectors) -> None; search(queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray] возвращает (scores, indices) формы (n_queries, k), scores по убыванию, indices int64; save(path) -> None; load(path) -> None; свойство size -> int. Наследники: NumpyIndex(); FaissIndex(index_type: str = "flat", nlist: int = 100, nprobe: int = 10), где index_type это "flat" или "ivf". Мера сходства: скалярное произведение нормированных векторов (равно косинусному сходству).
- src/engine.py: RecommendationEngine(items: pd.DataFrame, encoder: EmbeddingEncoder, index: BaseIndex). Метод recommend(query: str, k: int = 3, category: str | None = None) -> pd.DataFrame с колонками [rank, item_id, title, category, score, description]; rank начинается с 1.
- src/factory.py: build_engine(items_path: str | Path = "data/processed/items.csv", embeddings_path: str | Path | None = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_type: str = "faiss-flat") -> RecommendationEngine; index_type: "numpy", "faiss-flat" или "faiss-ivf". Если файла эмбеддингов нет, считает и кэширует.
- src/evaluator.py: Evaluator(engine: RecommendationEngine, eval_df: pd.DataFrame). Методы: recall_at_k(k: int) -> float; mrr() -> float; avg_latency_ms() -> float; run_all(ks=(1, 3, 10)) -> dict; per_query(k: int = 10) -> pd.DataFrame.
- src/app.py: интерфейс Streamlit, запуск: streamlit run src/app.py.

## Определения метрик
- Recall@k для одного запроса = |relevant ∩ top-k| / |relevant|; итог = среднее по запросам.
- MRR = среднее по запросам от 1 / (ранг первого релевантного в топ-10), либо 0, если не найден.
- Latency = время вызова engine.recommend() на один запрос в миллисекундах (perf_counter).
