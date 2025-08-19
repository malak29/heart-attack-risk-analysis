from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

class TrainingRequest(BaseModel):
    """Request schema for model training"""
    model_type: str = Field(..., regex='^(random_forest|gradient_boost|xgboost|ensemble)$',
                          description="Type of model to train")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, 
                                                     description="Custom hyperparameters")
    auto_tune: bool = Field(True, description="Enable hyperparameter tuning")
    validation_split: float = Field(0.2, ge=0.1, le=0.4, 
                                  description="Validation data split ratio")
    cv_folds: int = Field(5, ge=3, le=10, description="Cross-validation folds")
    optimize_metric: str = Field("roc_auc", 
                                regex='^(accuracy|precision|recall|f1|roc_auc)$',
                                description="Metric to optimize")
    max_training_time: Optional[int] = Field(None, ge=60, le=3600,
                                            description="Maximum training time in seconds")
    
    @validator('hyperparameters')
    def validate_hyperparameters(cls, v, values):
        """Validate hyperparameters based on model type"""
        if v and 'model_type' in values:
            model_type = values['model_type']
            
            # Define valid parameters for each model type
            valid_params = {
                'random_forest': ['n_estimators', 'max_depth', 'min_samples_split', 
                                'min_samples_leaf', 'max_features'],
                'gradient_boost': ['n_estimators', 'learning_rate', 'max_depth', 
                                 'min_samples_split', 'subsample'],
                'xgboost': ['n_estimators', 'max_depth', 'learning_rate', 
                          'subsample', 'colsample_bytree'],
                'ensemble': []  # Ensemble doesn't take direct hyperparameters
            }
            
            # Check if all provided parameters are valid
            invalid_params = set(v.keys()) - set(valid_params.get(model_type, []))
            if invalid_params:
                raise ValueError(f"Invalid parameters for {model_type}: {invalid_params}")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "random_forest",
                "auto_tune": True,
                "validation_split": 0.2,
                "cv_folds": 5,
                "optimize_metric": "roc_auc"
            }
        }

class TrainingResponse(BaseModel):
    """Response schema for model training"""
    model_version: str = Field(..., description="Trained model version")
    model_type: str = Field(..., description="Type of model trained")
    training_completed: bool = Field(..., description="Training completion status")
    training_time_seconds: float = Field(..., description="Training duration")
    metrics: 'ModelMetrics' = Field(..., description="Model performance metrics")
    best_parameters: Optional[Dict[str, Any]] = Field(None, 
                                                     description="Best hyperparameters found")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_version": "v20240101_120000",
                "model_type": "random_forest",
                "training_completed": True,
                "training_time_seconds": 45.3,
                "metrics": {
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.88,
                    "f1_score": 0.885,
                    "roc_auc": 0.95
                },
                "best_parameters": {
                    "n_estimators": 200,
                    "max_depth": 20,
                    "min_samples_split": 2
                }
            }
        }

class ModelMetrics(BaseModel):
    """Schema for model performance metrics"""
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Precision score")
    recall: float = Field(..., ge=0, le=1, description="Recall score")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")
    roc_auc: float = Field(..., ge=0, le=1, description="ROC AUC score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    train_score: Optional[float] = Field(None, ge=0, le=1, description="Training score")
    test_score: Optional[float] = Field(None, ge=0, le=1, description="Test score")
    cv_score_mean: Optional[float] = Field(None, ge=0, le=1, 
                                          description="Cross-validation mean score")
    cv_score_std: Optional[float] = Field(None, ge=0, 
                                         description="Cross-validation score std deviation")
    feature_importance: Optional[Dict[str, float]] = Field(None, 
                                                          description="Feature importance scores")

class ModelInfo(BaseModel):
    """Schema for model information"""
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type")
    status: str = Field(..., regex='^(active|inactive|archived)$', 
                       description="Model status")
    created_at: datetime = Field(..., description="Creation timestamp")
    metrics: ModelMetrics = Field(..., description="Model metrics")
    description: Optional[str] = Field(None, description="Model description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ModelComparison(BaseModel):
    """Schema for comparing multiple models"""
    models: List[ModelInfo] = Field(..., min_items=2, description="Models to compare")
    best_model: str = Field(..., description="Best model version")
    comparison_metric: str = Field(..., description="Metric used for comparison")
    improvement: float = Field(..., description="Improvement over baseline")