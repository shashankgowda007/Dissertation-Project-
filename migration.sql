-- migration.sql
-- Add missing columns to existing tables

-- Add columns to model_comparisons if they don't exist
ALTER TABLE model_comparisons ADD COLUMN clinical_confidence_level REAL DEFAULT 0.0;
ALTER TABLE model_comparisons ADD COLUMN agreement_level TEXT DEFAULT 'unknown';

-- Add columns to performance_metrics if they don't exist
ALTER TABLE performance_metrics ADD COLUMN total_processing_time REAL DEFAULT 0.0;
ALTER TABLE performance_metrics ADD COLUMN meets_time_target BOOLEAN DEFAULT FALSE;
ALTER TABLE performance_metrics ADD COLUMN cnn_vs_radiologist_sensitivity REAL DEFAULT 0.0;
ALTER TABLE performance_metrics ADD COLUMN cnn_vs_radiologist_specificity REAL DEFAULT 0.0;
ALTER TABLE performance_metrics ADD COLUMN vit_vs_radiologist_sensitivity REAL DEFAULT 0.0;
ALTER TABLE performance_metrics ADD COLUMN vit_vs_radiologist_specificity REAL DEFAULT 0.0;

-- Add columns to analysis_sessions if they don't exist
ALTER TABLE analysis_sessions ADD COLUMN processing_status TEXT DEFAULT 'pending';
ALTER TABLE analysis_sessions ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
