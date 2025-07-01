from sqlalchemy import create_engine, Column, Integer, String, Float, text
from sqlalchemy.orm import declarative_base

DATABASE_URL = 'sqlite:///utils/clinical_analysis.db'

Base = declarative_base()

class ModelPrediction(Base):
    __tablename__ = 'model_predictions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    model_type = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    raw_score = Column(Float)
    processing_time_seconds = Column(Float)
    model_sensitivity = Column(Float)
    model_specificity = Column(Float)

if __name__ == '__main__':
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    # Add score_difference column to model_comparisons if it does not exist
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(model_comparisons)"))
        columns = [row[1] for row in result]
        if 'score_difference' not in columns:
            conn.execute(text("ALTER TABLE model_comparisons ADD COLUMN score_difference FLOAT"))
            print("Added 'score_difference' column to model_comparisons table.")
        else:
            print("'score_difference' column already exists in model_comparisons table.")
    print('Table model_predictions has been created (if it did not exist).')