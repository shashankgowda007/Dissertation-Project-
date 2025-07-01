import sqlite3

# Quick fix to add all missing columns
def fix_table_columns(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List of columns to add
    columns_to_add = [
        "consensus_prediction TEXT",
        "consensus_confidence REAL", 
        "clinical_confidence_level TEXT",
        "agreement_level TEXT"
    ]
    
    for column in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE model_comparisons ADD COLUMN {column}")
            print(f"✅ Added column: {column}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"⚠️  Column already exists: {column}")
            else:
                print(f"❌ Error adding {column}: {e}")
    
    conn.commit()
    
    # Verify final structure
    cursor.execute("PRAGMA table_info(model_comparisons)")
    columns = cursor.fetchall()
    print("\n📋 Final table structure:")
    for col in columns:
        print(f"   {col[1]}: {col[2]}")
    
    conn.close()
    print("\n✅ Database update completed!")

# Replace with your database path
DATABASE_PATH = 'utils/clinical_analysis.db'  # Update this path!

if __name__ == "__main__":
    fix_table_columns(DATABASE_PATH)