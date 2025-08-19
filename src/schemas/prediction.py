from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    patient_data: Dict[str, Any] = Field(..., description="Patient data for prediction")
    return_features: bool = Field(False, description="Return engineered features")
    include_explanation: bool = Field(False, description="Include prediction explanation")

class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    patient_id: str = Field(..., description="Unique patient identifier")
    risk_score: float = Field(..., ge=0, le=1, description="Risk probability (0-1)")
    risk_level: str = Field(..., description="Risk level category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Personalized recommendations")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    
    # Optional fields
    engineered_features: Optional[Dict[str, Any]] = Field(None, 
                                                         description="Engineered features used")
    explanation: Optional[Dict[str, float]] = Field(None, 
                                                   description="Feature importance for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "abc123def456",
                "risk_score": 0.72,
                "risk_level": "High",
                "confidence": 0.85,
                "recommendations": [
                    "⚠️ Immediate consultation with a cardiologist recommended",
                    "🚭 Enroll in a smoking cessation program",
                    "🏃 Increase physical activity to 150+ minutes/week"
                ],
                "model_version": "v20240101_120000",
                "timestamp": "2024-01-01T12:00:00",
                "response_time_ms": 45.2
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    patients: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000,
                                          description="List of patient data")
    return_features: bool = Field(False, description="Return engineered features")
    parallel_processing: bool = Field(True, description="Use parallel processing")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_patients: int = Field(..., description="Total number of patients processed")
    processing_time_seconds: float = Field(..., description="Total processing time")
    failed_predictions: List[Dict[str, str]] = Field(default=[], 
                                                    description="Failed predictions with errors")
    batch_id: str = Field(..., description="Unique batch identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "patient_id": "abc123def456",
                        "risk_score": 0.72,
                        "risk_level": "High",
                        "confidence": 0.85,
                        "recommendations": ["Consultation recommended"],
                        "model_version": "v20240101_120000",
                        "timestamp": "2024-01-01T12:00:00",
                        "response_time_ms": 45.2
                    }
                ],
                "total_patients": 1,
                "processing_time_seconds": 0.5,
                "failed_predictions": [],
                "batch_id": "batch_20240101_120000"
            }
        }

class PredictionHistory(BaseModel):
    """Schema for prediction history"""
    patient_id: str
    predictions: List[PredictionResponse]
    trend: str = Field(..., description="Risk trend over time (increasing/stable/decreasing)")
    
class RiskAssessment(BaseModel):
    """Detailed risk assessment response"""
    overall_risk: float = Field(..., ge=0, le=1)
    risk_factors: Dict[str, float] = Field(..., description="Individual risk factor contributions")
    protective_factors: List[str] = Field(..., description="Factors reducing risk")
    risk_trajectory: str = Field(..., description="Expected risk trajectory")
    intervention_priority: List[str] = Field(..., description="Prioritized interventions")