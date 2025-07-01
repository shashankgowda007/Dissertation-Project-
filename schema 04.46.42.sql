-- schema.sql
CREATE TABLE IF NOT EXISTS analysis_sessions (
    session_id TEXT PRIMARY KEY,
    image_filename TEXT NOT NULL,
    image_size_mb REAL,
    image_format TEXT,
    image_dimensions TEXT,
    processing_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    model_type TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_score REAL,
    processing_time_seconds REAL,
    model_sensitivity REAL,
    model_specificity REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS model_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    prediction_agreement BOOLEAN,
    confidence_difference REAL,
    score_difference REAL,
    consensus_prediction TEXT,
    consensus_confidence REAL,
    clinical_confidence_level REAL,
    agreement_level TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    total_processing_time REAL,
    meets_time_target BOOLEAN,
    cnn_vs_radiologist_sensitivity REAL,
    cnn_vs_radiologist_specificity REAL,
    vit_vs_radiologist_sensitivity REAL,
    vit_vs_radiologist_specificity REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS clinical_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    report_content TEXT NOT NULL,
    report_type TEXT DEFAULT 'standard',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)
);
