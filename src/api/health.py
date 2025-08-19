from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import platform
from pathlib import Path
from src.config import Settings
from src.model_registry import ModelRegistry
from src.monitoring import ModelMonitor

router = APIRouter(prefix="/health", tags=["Health & Monitoring"])

settings = Settings()
model_registry = ModelRegistry()
monitor = ModelMonitor()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "Heart Attack Prediction API"
    }

@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with system information"""
    
    # Check model status
    try:
        current_model = model_registry.get_current_model()
        model_status = "loaded" if current_model else "not_loaded"
        model_version = model_registry.current_version
    except Exception as e:
        model_status = "error"
        model_version = None
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check critical paths
    paths_status = {
        "models": Path(settings.model_dir).exists(),
        "data": Path(settings.data_dir).exists(),
        "logs": Path("logs").exists()
    }
    
    return {
        "status": "healthy" if model_status != "error" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "status": model_status,
            "version": model_version,
            "total_models": len(model_registry.models)
        },
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        },
        "paths": paths_status,
        "uptime_seconds": (datetime.now() - datetime.now()).total_seconds()  # Would need actual start time
    }

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint"""
    
    # Check if model is loaded
    try:
        model = model_registry.get_current_model()
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")
    
    return {"ready": True, "timestamp": datetime.now().isoformat()}

@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint"""
    return {"alive": True, "timestamp": datetime.now().isoformat()}

@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get application metrics for monitoring"""
    
    # Get performance summary
    perf_summary = monitor.get_performance_summary(window_hours=24)
    
    # Get model metrics
    model_metrics = {}
    if model_registry.current_version:
        model_info = model_registry.get_model_info(model_registry.current_version)
        if model_info:
            model_metrics = model_info.get('metrics', {})
    
    return {
        "timestamp": datetime.now().isoformat(),
        "predictions": perf_summary,
        "model_metrics": model_metrics,
        "active_model": model_registry.current_version
    }

@router.get("/dependencies")
async def check_dependencies() -> Dict[str, Any]:
    """Check status of external dependencies"""
    
    dependencies = {}
    
    # Check database connection
    try:
        from src.database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        dependencies["database"] = "healthy"
    except Exception as e:
        dependencies["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis if configured
    if settings.redis_url:
        try:
            import redis
            r = redis.from_url(settings.redis_url)
            r.ping()
            dependencies["redis"] = "healthy"
        except Exception as e:
            dependencies["redis"] = f"unhealthy: {str(e)}"
    
    # Check MLflow if configured
    if settings.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            # Simple check - would need actual MLflow client check
            dependencies["mlflow"] = "healthy"
        except Exception as e:
            dependencies["mlflow"] = f"unhealthy: {str(e)}"
    
    all_healthy = all(v == "healthy" for v in dependencies.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "dependencies": dependencies,
        "timestamp": datetime.now().isoformat()
    }