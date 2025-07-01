-- Migration script to add missing columns to performance_metrics table

BEGIN TRANSACTION;

-- Add total_processing_time column if it does not exist
ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS total_processing_time REAL DEFAULT 0;

-- Add meets_time_target column if it does not exist
ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS meets_time_target BOOLEAN DEFAULT FALSE;

COMMIT;
