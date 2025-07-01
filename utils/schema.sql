-- SQL schema for SQLite database tables used in the pneumonia detection app

CREATE TABLE IF NOT EXISTS analysis_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    image_filename TEXT NOT NULL,
    image_size_mb REAL NOT NULL,
    image_format TEXT NOT NULL,
    image_dimensions TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS model_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cnn_prediction TEXT,
    vit_prediction TEXT,
    prediction_agreement BOOLEAN,
    confidence_difference REAL,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cnn_processing_time REAL,
    vit_processing_time REAL,
    total_processing_time REAL,
    meets_time_target BOOLEAN,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
);
