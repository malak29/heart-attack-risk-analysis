from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from datetime import datetime
from typing import Optional
import json
from src.config import Settings

settings = Settings()

# Create engine
if settings.database_url:
    if "sqlite" in settings.database_url:
        engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
    else:
        engine = create_engine(settings.database_url, echo=False)
else:
    # In-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Database Models
class Prediction(Base):
    """Model for storing predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    input_data = Column(JSON)
    risk_score = Column(Float)
    risk_level = Column(String)
    confidence = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class ModelInfo(Base):
    """Model for storing model information"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    model_type = Column(String)
    metrics = Column(JSON)
    parameters = Column(JSON)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class TrainingJob(Base):
    """Model for tracking training jobs"""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    status = Column(String)  # pending, running, completed, failed
    model_type = Column(String)
    parameters = Column(JSON)
    metrics = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency for getting DB session
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()