from src.api import health, prediction, training, model_management

__all__ = [
    'health',
    'prediction', 
    'training',
    'model_management'
]

# API version
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"