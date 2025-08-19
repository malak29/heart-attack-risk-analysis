from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.schemas.patient import PatientData, PatientBatchData
from src.schemas.prediction import (
    PredictionResponse, BatchPredictionResponse,
    PredictionRequest, BatchPredictionRequest, RiskAssessment
)
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_registry import ModelRegistry
from src.monitoring import ModelMonitor
from src.utils import (
    generate_patient_id, calculate_risk_score,
    determine_risk_level, generate_recommendations
)
from src.config import Settings
from src.exceptions import ModelNotFoundError, PredictionError

router = APIRouter(prefix="/predict", tags=["Predictions"])

settings = Settings()
data_processor = DataProcessor()
feature_engineer = FeatureEngineer()
model_registry = ModelRegistry()
monitor = ModelMonitor()

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

@router.post("/", response_model=PredictionResponse)
async def predict_single(
    patient_data: PatientData,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """Make a prediction for a single patient"""
    
    start_time = time.time()
    
    try:
        # Get current model
        model = model_registry.get_current_model()
        if not model:
            raise ModelNotFoundError("No active model available")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([patient_data.dict()])
        
        # Process and engineer features
        processed_df = data_processor.transform(input_df)
        features_df = feature_engineer.create_features(processed_df)
        
        # Make prediction
        risk_proba = model.predict_proba(features_df)[0]
        risk_score = calculate_risk_score(risk_proba)
        risk_level = determine_risk_level(risk_score)
        
        # Generate patient ID
        patient_id = generate_patient_id(patient_data.dict())
        
        # Generate recommendations
        recommendations = generate_recommendations(
            patient_data.dict(),
            risk_level
        )
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # ms
        
        # Log prediction in background
        background_tasks.add_task(
            monitor.log_prediction,
            patient_data.dict(),
            risk_score,
            model_registry.current_version,
            response_time / 1000
        )
        
        return PredictionResponse(
            patient_id=patient_id,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=max(risk_proba),
            recommendations=recommendations,
            model_version=model_registry.current_version,
            timestamp=datetime.now(),
            response_time_ms=response_time
        )
        
    except ModelNotFoundError:
        raise HTTPException(status_code=503, detail="No model available for prediction")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
) -> BatchPredictionResponse:
    """Make predictions for multiple patients"""
    
    start_time = time.time()
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Get current model
        model = model_registry.get_current_model()
        if not model:
            raise ModelNotFoundError("No active model available")
        
        predictions = []
        failed_predictions = []
        
        # Process each patient
        for i, patient_dict in enumerate(request.patients):
            try:
                # Create PatientData object for validation
                patient = PatientData(**patient_dict)
                
                # Process individually
                input_df = pd.DataFrame([patient.dict()])
                processed_df = data_processor.transform(input_df)
                features_df = feature_engineer.create_features(processed_df)
                
                # Predict
                risk_proba = model.predict_proba(features_df)[0]
                risk_score = calculate_risk_score(risk_proba)
                risk_level = determine_risk_level(risk_score)
                
                # Create response
                predictions.append(PredictionResponse(
                    patient_id=generate_patient_id(patient.dict()),
                    risk_score=risk_score,
                    risk_level=risk_level,
                    confidence=max(risk_proba),
                    recommendations=generate_recommendations(patient.dict(), risk_level),
                    model_version=model_registry.current_version,
                    timestamp=datetime.now(),
                    response_time_ms=0  # Not tracked for batch
                ))
                
            except Exception as e:
                failed_predictions.append({
                    "index": i,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_patients=len(request.patients),
            processing_time_seconds=processing_time,
            failed_predictions=failed_predictions,
            batch_id=batch_id
        )
        
    except ModelNotFoundError:
        raise HTTPException(status_code=503, detail="No model available for prediction")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.post("/assess", response_model=RiskAssessment)
async def assess_risk(patient_data: PatientData) -> RiskAssessment:
    """Get detailed risk assessment for a patient"""
    
    try:
        # Get current model
        model = model_registry.get_current_model()
        if not model:
            raise ModelNotFoundError("No active model available")
        
        # Process data
        input_df = pd.DataFrame([patient_data.dict()])
        processed_df = data_processor.transform(input_df)
        features_df = feature_engineer.create_features(processed_df)
        
        # Get prediction
        risk_proba = model.predict_proba(features_df)[0]
        risk_score = calculate_risk_score(risk_proba)
        
        # Get feature importance if available
        risk_factors = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, col in enumerate(features_df.columns):
                if importances[i] > 0.01:  # Only significant factors
                    risk_factors[col] = float(importances[i])
        
        # Identify protective factors
        protective_factors = []
        if not patient_data.smoking:
            protective_factors.append("Non-smoker")
        if patient_data.exercise_hours_per_week >= 2.5:
            protective_factors.append("Regular exercise")
        if patient_data.diet == "Healthy":
            protective_factors.append("Healthy diet")
        
        # Determine trajectory
        if risk_score < 0.3:
            trajectory = "stable_low"
        elif risk_score < 0.6:
            trajectory = "moderate_watch"
        else:
            trajectory = "high_intervention_needed"
        
        # Priority interventions
        interventions = []
        if patient_data.smoking:
            interventions.append("Smoking cessation")
        if patient_data.bmi > 30:
            interventions.append("Weight management")
        if patient_data.cholesterol > 240:
            interventions.append("Cholesterol control")
        
        return RiskAssessment(
            overall_risk=risk_score,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            risk_trajectory=trajectory,
            intervention_priority=interventions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")