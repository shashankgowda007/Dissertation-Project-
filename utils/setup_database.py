import sqlite3
import os

def setup_database(db_path):
    """Ensure all required columns exist"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List of required columns for each table
    required_columns = {
        'model_comparisons': [
            ('clinical_confidence_level', 'REAL DEFAULT 0.0'),
            ('prediction_agreement', 'BOOLEAN DEFAULT FALSE'),
            ('confidence_difference', 'REAL DEFAULT 0.0')
        ],
        'performance_metrics': [
            ('total_processing_time', 'REAL DEFAULT 0.0'),
            ('meets_time_target', 'BOOLEAN DEFAULT FALSE')
        ]
    }
    
    for table_name, columns in required_columns.items():
        for column_name, column_def in columns:
            try:
                cursor.execute(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN {column_name} {column_def}
                """)
                print(f"Added {column_name} to {table_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"Column {column_name} already exists in {table_name}")
                else:
                    print(f"Error adding {column_name} to {table_name}: {e}")
    
    conn.commit()
    conn.close()
    print("Database setup complete!")

# Run this to fix your database
if __name__ == "__main__":
    db_path = input("Enter your database path: ").strip()
    if os.path.exists(db_path):
        setup_database(db_path)
    else:
        print(f"Database file not found: {db_path}")
