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