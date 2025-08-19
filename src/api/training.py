from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends
from typing import Dict, Any, Optional
import pandas as pd
import asyncio
from datetime import datetime
import uuid
from pathlib import Path
import joblib

from src.schemas.training import TrainingRequest, TrainingResponse, ModelMetrics
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_registry import ModelRegistry
from src.config import Settings
from src.database import get_db, TrainingJob
from sqlalchemy.orm import Session

router = APIRouter(prefix="/train", tags=["Model Training"])

settings = Settings()
data_processor = DataProcessor()
feature_engineer = FeatureEngineer()
model_trainer = ModelTrainer()
model_registry = ModelRegistry()

# Store active training jobs
active_jobs = {}

@router.post("/", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> TrainingResponse:
    """Train a new model with specified parameters"""
    
    start_time = datetime.now()
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    try:
        # Load training data
        train_data_path = settings.processed_dir / "train_data.csv"
        if not train_data_path.exists():
            raise FileNotFoundError("Training data not found. Please upload data first.")
        
        df = pd.read_csv(train_data_path)
        
        # Separate features and target
        if 'heart_attack_risk' not in df.columns:
            raise ValueError("Target column 'heart_attack_risk' not found in data")
        
        X = df.drop('heart_attack_risk', axis=1)
        y = df['heart_attack_risk']
        
        # Create training job record
        training_job = TrainingJob(
            job_id=job_id,
            status="running",
            model_type=request.model_type,
            parameters=request.dict(),
            started_at=start_time
        )
        db.add(training_job)
        db.commit()
        
        # Train model based on request
        if request.auto_tune:
            model, metrics = model_trainer.train_with_hyperparameter_tuning(
                X, y,
                model_type=request.model_type,
                validation_split=request.validation_split,
                cv_folds=request.cv_folds
            )
            best_params = metrics.get('best_params', {})
        else:
            model, metrics = model_trainer.train_model(
                X, y,
                model_type=request.model_type,
                hyperparameters=request.hyperparameters,
                validation_split=request.validation_split
            )
            best_params = request.hyperparameters
        
        # Register model
        model_version = model_registry.register_model(
            model=model,
            metrics=metrics,
            model_type=request.model_type,
            description=f"Trained via API - Job {job_id}",
            metadata={"job_id": job_id, "request": request.dict()}
        )
        
        # Update training job
        training_job.status = "completed"
        training_job.metrics = metrics
        training_job.completed_at = datetime.now()
        db.commit()
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        return TrainingResponse(
            model_version=model_version,
            model_type=request.model_type,
            training_completed=True,
            training_time_seconds=training_time,
            metrics=ModelMetrics(**metrics),
            best_parameters=best_params
        )
        
    except Exception as e:
        # Update job status on failure
        if 'training_job' in locals():
            training_job.status = "failed"
            training_job.error_message = str(e)
            training_job.completed_at = datetime.now()
            db.commit()
        
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/upload-data")
async def upload_training_data(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """Upload and process training data"""
    
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx')):
            raise ValueError("Only CSV and Excel files are supported")
        
        # Save uploaded file
        upload_path = settings.upload_dir / file.filename
        content = await file.read()
        
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Process in background
        background_tasks.add_task(process_training_data, str(upload_path))
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "status": "processing",
            "path": str(upload_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

async def process_training_data(file_path: str):
    """Background task to process uploaded training data"""
    
    try:
        # Load data
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Process data
        df_processed = data_processor.fit_transform(df)
        
        # Engineer features
        df_features = feature_engineer.create_features(df_processed)
        
        # Save processed data
        output_path = settings.processed_dir / "train_data.csv"
        df_features.to_csv(output_path, index=False)
        
        print(f"Training data processed and saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing training data: {str(e)}")

@router.get("/jobs")
async def list_training_jobs(
    db: Session = Depends(get_db),
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """List all training jobs"""
    
    jobs = db.query(TrainingJob).offset(offset).limit(limit).all()
    total = db.query(TrainingJob).count()
    
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "model_type": job.model_type,
                "started_at": job.started_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "metrics": job.metrics
            }
            for job in jobs
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }

@router.get("/jobs/{job_id}")
async def get_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get details of a specific training job"""
    
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "model_type": job.model_type,
        "parameters": job.parameters,
        "metrics": job.metrics,
        "error_message": job.error_message,
        "started_at": job.started_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }

@router.post("/retrain")
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Trigger model retraining with current best parameters"""
    
    try:
        # Get current model info
        current_model_info = model_registry.get_model_info(model_registry.current_version)
        if not current_model_info:
            raise ValueError("No current model found")
        
        # Use same model type and parameters
        model_type = current_model_info['type']
        best_params = current_model_info.get('metadata', {}).get('best_params', {})
        
        # Create training request
        request = TrainingRequest(
            model_type=model_type,
            hyperparameters=best_params,
            auto_tune=False
        )
        
        # Trigger training
        response = await train_model(request, background_tasks, db)
        
        return {
            "message": "Retraining triggered",
            "new_version": response.model_version,
            "model_type": model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# ============== FILE 2: src/api/model_management.py ==============
"""
Model management endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil
from pathlib import Path

from src.schemas.training import ModelInfo, ModelComparison, ModelMetrics
from src.model_registry import ModelRegistry
from src.monitoring import ModelMonitor
from src.config import Settings
from src.database import get_db, ModelInfo as DBModelInfo
from sqlalchemy.orm import Session

router = APIRouter(prefix="/models", tags=["Model Management"])

settings = Settings()
model_registry = ModelRegistry()
monitor = ModelMonitor()

@router.get("/", response_model=List[ModelInfo])
async def list_models(
    status: Optional[str] = None,
    limit: int = 10
) -> List[ModelInfo]:
    """List all registered models"""
    
    models = model_registry.list_models()
    
    # Filter by status if provided
    if status:
        models = [m for m in models if m['status'] == status]
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limit results
    models = models[:limit]
    
    # Convert to response schema
    return [
        ModelInfo(
            version=m['version'],
            type=m['type'],
            status=m['status'],
            created_at=datetime.fromisoformat(m['created_at']),
            metrics=ModelMetrics(**m['metrics']),
            description=m.get('description'),
            metadata=m.get('metadata')
        )
        for m in models
    ]

@router.get("/current")
async def get_current_model() -> ModelInfo:
    """Get information about the currently active model"""
    
    if not model_registry.current_version:
        raise HTTPException(status_code=404, detail="No active model")
    
    model_info = model_registry.get_model_info(model_registry.current_version)
    
    if not model_info:
        raise HTTPException(status_code=404, detail="Model information not found")
    
    return ModelInfo(
        version=model_info['version'],
        type=model_info['type'],
        status=model_info['status'],
        created_at=datetime.fromisoformat(model_info['created_at']),
        metrics=ModelMetrics(**model_info['metrics']),
        description=model_info.get('description'),
        metadata=model_info.get('metadata')
    )

@router.get("/{version}")
async def get_model(version: str) -> ModelInfo:
    """Get information about a specific model version"""
    
    model_info = model_registry.get_model_info(version)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {version} not found")
    
    return ModelInfo(
        version=model_info['version'],
        type=model_info['type'],
        status=model_info['status'],
        created_at=datetime.fromisoformat(model_info['created_at']),
        metrics=ModelMetrics(**model_info['metrics']),
        description=model_info.get('description'),
        metadata=model_info.get('metadata')
    )

@router.post("/{version}/activate")
async def activate_model(version: str) -> Dict[str, Any]:
    """Activate a specific model version"""
    
    try:
        model_registry.activate_model(version)
        
        # Log the activation
        monitor.log_metrics(version, {"event": "model_activated"})
        
        return {
            "message": f"Model {version} activated successfully",
            "version": version,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation failed: {str(e)}")

@router.delete("/{version}")
async def delete_model(version: str) -> Dict[str, Any]:
    """Delete a specific model version"""
    
    try:
        model_registry.delete_model(version)
        
        return {
            "message": f"Model {version} deleted successfully",
            "version": version,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.post("/compare", response_model=ModelComparison)
async def compare_models(version1: str, version2: str) -> ModelComparison:
    """Compare two model versions"""
    
    # Get model information
    model1 = model_registry.get_model_info(version1)
    model2 = model_registry.get_model_info(version2)
    
    if not model1 or not model2:
        raise HTTPException(status_code=404, detail="One or both models not found")
    
    # Compare based on ROC AUC
    metric = "roc_auc"
    score1 = model1['metrics'].get(metric, 0)
    score2 = model2['metrics'].get(metric, 0)
    
    best_version = version1 if score1 >= score2 else version2
    improvement = abs(score1 - score2)
    
    return ModelComparison(
        models=[
            ModelInfo(
                version=model1['version'],
                type=model1['type'],
                status=model1['status'],
                created_at=datetime.fromisoformat(model1['created_at']),
                metrics=ModelMetrics(**model1['metrics'])
            ),
            ModelInfo(
                version=model2['version'],
                type=model2['type'],
                status=model2['status'],
                created_at=datetime.fromisoformat(model2['created_at']),
                metrics=ModelMetrics(**model2['metrics'])
            )
        ],
        best_model=best_version,
        comparison_metric=metric,
        improvement=improvement
    )

@router.post("/cleanup")
async def cleanup_old_models(keep_last_n: int = 5) -> Dict[str, Any]:
    """Clean up old models, keeping only the last n versions"""
    
    try:
        initial_count = len(model_registry.models)
        model_registry.cleanup_old_models(keep_last_n)
        final_count = len(model_registry.models)
        
        return {
            "message": "Cleanup completed",
            "models_deleted": initial_count - final_count,
            "models_remaining": final_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/{version}/download")
async def download_model(version: str) -> Dict[str, Any]:
    """Get download URL for a model (in production, would return presigned URL)"""
    
    model_info = model_registry.get_model_info(version)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {version} not found")
    
    # In production, generate a presigned URL for S3/GCS
    # For now, return the local path
    return {
        "version": version,
        "path": model_info['path'],
        "message": "In production, this would return a presigned download URL"
    }