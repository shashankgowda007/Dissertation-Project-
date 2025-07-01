import os
import uuid
import time
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine, text
import streamlit as st

class DatabaseManager:
    """
    Database management for clinical analysis sessions and results
    Handles storage of analysis results, model performance tracking, and audit logs
    """
    
    def __init__(self):
        # Use local SQLite database file
        db_path = os.path.join(os.path.dirname(__file__), 'clinical_analysis.db')
        self.engine = create_engine(f'sqlite:///{db_path}')
        self._initialize_database()

    def _initialize_database(self):
        """Create tables if they do not exist"""
        with self.engine.begin() as conn:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            # Split the schema into individual statements and execute separately
            for statement in schema_sql.split(';'):
                stmt = statement.strip()
                if stmt:
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        # Log or ignore errors for existing tables or columns
                        pass
            # Run migration script to add missing columns
            migration_path = os.path.join(os.path.dirname(__file__), 'migration.sql')
            with open(migration_path, 'r') as f:
                migration_sql = f.read()
            for statement in migration_sql.split(';'):
                stmt = statement.strip()
                if stmt:
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        # Log or ignore errors for existing columns
                        pass
            # Refresh metadata cache to recognize new columns
            conn.execute(text("PRAGMA writable_schema = 1"))
            conn.execute(text("PRAGMA writable_schema = 0"))
    
    def create_analysis_session(self, image_filename, image_size_mb, image_format, image_dimensions):
        """Create a new analysis session and return session ID"""
        session_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO analysis_sessions 
        (session_id, image_filename, image_size_mb, image_format, image_dimensions)
        VALUES (:session_id, :filename, :size, :format, :dimensions)
        """
        
        with self.engine.begin() as conn:
            conn.execute(
                text(query),
                {
                    'session_id': session_id,
                    'filename': image_filename,
                    'size': image_size_mb,
                    'format': image_format,
                    'dimensions': image_dimensions
                }
            )
        
        return session_id
    
    def save_model_prediction(self, session_id, model_type, prediction_result, processing_time):
        """Save individual model prediction results"""
        query = """
        INSERT INTO model_predictions 
        (session_id, model_type, prediction, confidence, raw_score, processing_time_seconds, 
         model_sensitivity, model_specificity)
        VALUES (:session_id, :model_type, :prediction, :confidence, :raw_score, :processing_time,
                :sensitivity, :specificity)
        """
        
        # Convert all numeric values to Python native types to avoid NumPy type issues
        confidence_val = float(prediction_result['confidence'])
        raw_score_val = float(prediction_result.get('raw_score', prediction_result['confidence']))
        processing_time_val = float(processing_time)
        
        # Get sensitivity and specificity from model info or use defaults
        model_info = prediction_result.get('model_info', {})
        sensitivity_val = float(model_info.get('sensitivity', 0.94 if model_type == 'CNN' else 0.95))
        specificity_val = float(model_info.get('specificity', 0.918 if model_type == 'CNN' else 0.925))
        
        with self.engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'session_id': session_id,
                    'model_type': model_type,
                    'prediction': prediction_result['prediction'],
                    'confidence': confidence_val,
                    'raw_score': raw_score_val,
                    'processing_time': processing_time_val,
                    'sensitivity': sensitivity_val,
                    'specificity': specificity_val
                }
            )
            conn.commit()
    
    def save_model_comparison(self, session_id, cnn_result, vit_result, metrics):
        """Save model comparison results"""
        agreement_data = metrics['agreement_analysis']
        confidence_data = metrics['confidence_analysis']['clinical_confidence_assessment']
        
        query = """
        INSERT INTO model_comparisons 
        (session_id, prediction_agreement, confidence_difference, score_difference,
         consensus_prediction, consensus_confidence, clinical_confidence_level, agreement_level)
        VALUES (:session_id, :agreement, :conf_diff, :score_diff, :consensus_pred, 
                :consensus_conf, :clinical_conf, :agreement_level)
        """
        
        with self.engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'session_id': session_id,
                    'agreement': agreement_data['prediction_agreement'],
                    'conf_diff': float(agreement_data['confidence_difference']),
                    'score_diff': float(abs(cnn_result['raw_score'] - vit_result['raw_score'])),
                    'consensus_pred': agreement_data['consensus_prediction']['prediction'],
                    'consensus_conf': float(agreement_data['consensus_prediction']['confidence']),
                    'clinical_conf': confidence_data['overall_confidence_level'],
                    'agreement_level': agreement_data['agreement_level']
                }
            )
            conn.commit()
    
    def save_performance_metrics(self, session_id, cnn_time, vit_time, metrics):
        """Save performance metrics"""
        benchmarks = metrics['clinical_benchmarks']
        efficiency = metrics['efficiency_metrics']
        
        query = """
        INSERT INTO performance_metrics 
        (session_id, total_processing_time, meets_time_target,
         cnn_vs_radiologist_sensitivity, cnn_vs_radiologist_specificity,
         vit_vs_radiologist_sensitivity, vit_vs_radiologist_specificity)
        VALUES (:session_id, :total_time, :meets_target, :cnn_sens, :cnn_spec, :vit_sens, :vit_spec)
        """
        
        total_time = cnn_time + vit_time
        
        with self.engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'session_id': session_id,
                    'total_time': total_time,
                    'meets_target': total_time <= 15.0,
                    'cnn_sens': benchmarks['cnn_vs_targets']['sensitivity_vs_radiologist'],
                    'cnn_spec': benchmarks['cnn_vs_targets']['specificity_vs_radiologist'],
                    'vit_sens': benchmarks['vit_vs_targets']['sensitivity_vs_radiologist'],
                    'vit_spec': benchmarks['vit_vs_targets']['specificity_vs_radiologist']
                }
            )
            conn.commit()
    
    def save_clinical_report(self, session_id, report_content, report_type='standard'):
        """Save generated clinical report"""
        query = """
        INSERT INTO clinical_reports (session_id, report_content, report_type)
        VALUES (:session_id, :content, :type)
        """
        
        with self.engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'session_id': session_id,
                    'content': report_content,
                    'type': report_type
                }
            )
            conn.commit()
    
    def update_session_status(self, session_id, status):
        """Update analysis session processing status"""
        query = """
        UPDATE analysis_sessions 
        SET processing_status = :status, updated_at = CURRENT_TIMESTAMP
        WHERE session_id = :session_id
        """
        
        with self.engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    'session_id': session_id,
                    'status': status
                }
            )
            conn.commit()
    
    def get_recent_analyses(self, limit=10):
        """Get recent analysis sessions for dashboard"""
        query = """
        SELECT 
            s.session_id,
            s.created_at,
            s.image_filename,
            s.processing_status,
            c.prediction_agreement,
            pm.total_processing_time
        FROM analysis_sessions s
        LEFT JOIN model_comparisons c ON s.session_id = c.session_id
        LEFT JOIN performance_metrics pm ON s.session_id = pm.session_id
        ORDER BY s.created_at DESC
        LIMIT :limit
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'limit': limit})
            return result.fetchall()
    
    def get_performance_statistics(self):
        """Get aggregate performance statistics"""
        query = """
        SELECT 
            COUNT(*) as total_analyses,
            AVG(pm.total_processing_time) as avg_processing_time,
            CASE 
                WHEN COUNT(*) > 0 THEN COUNT(CASE WHEN c.prediction_agreement THEN 1 END) * 100.0 / COUNT(*)
                ELSE 0
            END as agreement_rate,
            CASE 
                WHEN COUNT(*) > 0 THEN COUNT(CASE WHEN pm.meets_time_target THEN 1 END) * 100.0 / COUNT(*)
                ELSE 0
            END as time_target_rate,
            AVG(c.confidence_difference) as avg_confidence_diff
        FROM analysis_sessions s
        LEFT JOIN model_comparisons c ON s.session_id = c.session_id
        LEFT JOIN performance_metrics pm ON s.session_id = pm.session_id
        WHERE s.processing_status = 'completed'
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.fetchone()
    
    def get_model_performance_trends(self, days=30):
        """Get model performance trends over time"""
        query = """
        SELECT 
            DATE(s.created_at) as analysis_date,
            AVG(CASE WHEN mp.model_type = 'CNN' THEN mp.confidence END) as avg_cnn_confidence,
            AVG(CASE WHEN mp.model_type = 'VIT' THEN mp.confidence END) as avg_vit_confidence,
            AVG(pm.total_processing_time) as avg_processing_time,
            COUNT(*) as daily_count
        FROM analysis_sessions s
        JOIN model_predictions mp ON s.session_id = mp.session_id
        JOIN performance_metrics pm ON s.session_id = pm.session_id
        WHERE s.created_at >= CURRENT_DATE - INTERVAL '%s days'
        AND s.processing_status = 'completed'
        GROUP BY DATE(s.created_at)
        ORDER BY analysis_date DESC
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query % days))
            return result.fetchall()
    
    def save_complete_analysis(self, session_id, cnn_result, vit_result, cnn_time, vit_time, metrics, report_content):
        """Save complete analysis results in a single transaction"""
        try:
            # Save model predictions
            self.save_model_prediction(session_id, 'CNN', cnn_result, cnn_time)
            self.save_model_prediction(session_id, 'VIT', vit_result, vit_time)
            
            # Save comparison results
            self.save_model_comparison(session_id, cnn_result, vit_result, metrics)
            
            # Save performance metrics
            self.save_performance_metrics(session_id, cnn_time, vit_time, metrics)
            
            # Save clinical report
            self.save_clinical_report(session_id, report_content)
            
            # Update session status
            self.update_session_status(session_id, 'completed')
            
            return True
            
        except Exception as e:
            # Update session status to failed
            self.update_session_status(session_id, 'failed')
            raise e