-- SQLite schema for the prototype
CREATE TABLE articles (
    article_id TEXT,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    year INTEGER,
    journal TEXT,
    quartile TEXT,
    subject_area TEXT,
    citations INTEGER,
    source TEXT
);
CREATE TABLE analysis_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    dataset_name TEXT,
    article_count INTEGER,
    pearson_r REAL,
    spearman_r REAL,
    notes TEXT
);
