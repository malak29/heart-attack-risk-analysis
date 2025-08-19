import joblib
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional
import shutil

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manages model versions, storage, and retrieval"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self.current_version = None
        self.models = {}
        self._load_registry()
    
    def register_model(
        self,
        model: Any,
        metrics: Dict,
        model_type: str,
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> str:
        """Register a new model version"""
        
        # Generate version
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        model_path = self.models_dir / f"model_{version}.pkl"
        joblib.dump(model, model_path)
        
        # Create model info
        model_info = {
            'version': version,
            'type': model_type,
            'metrics': metrics,
            'created_at': datetime.now().isoformat(),
            'description': description,
            'path': str(model_path),
            'status': 'inactive',
            'metadata': metadata or {}
        }
        
        # Add to registry
        self.models[version] = model_info
        self._save_registry()
        
        logger.info(f"Model {version} registered successfully")
        return version
    
    def activate_model(self, version: str):
        """Activate a specific model version"""
        if version not in self.models:
            raise ValueError(f"Model {version} not found")
        
        # Deactivate current model
        if self.current_version:
            self.models[self.current_version]['status'] = 'inactive'
        
        # Activate new model
        self.models[version]['status'] = 'active'
        self.current_version = version
        self._save_registry()
        
        logger.info(f"Model {version} activated")
    
    def get_current_model(self) -> Optional[Any]:
        """Get the currently active model"""
        if not self.current_version:
            return None
        
        model_path = self.models[self.current_version]['path']
        return joblib.load(model_path)
    
    def get_model(self, version: str) -> Optional[Any]:
        """Get a specific model version"""
        if version not in self.models:
            return None
        
        model_path = self.models[version]['path']
        return joblib.load(model_path)
    
    def get_model_info(self, version: str) -> Optional[Dict]:
        """Get information about a specific model"""
        return self.models.get(version)
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        return list(self.models.values())
    
    def delete_model(self, version: str):
        """Delete a model version"""
        if version not in self.models:
            raise ValueError(f"Model {version} not found")
        
        if version == self.current_version:
            raise ValueError("Cannot delete active model")
        
        # Delete model file
        model_path = Path(self.models[version]['path'])
        if model_path.exists():
            model_path.unlink()
        
        # Remove from registry
        del self.models[version]
        self._save_registry()
        
        logger.info(f"Model {version} deleted")
    
    def cleanup_old_models(self, keep_last_n: int = 5):
        """Clean up old models, keeping only the last n versions"""
        # Sort models by creation time
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )
        
        # Keep active model and last n models
        models_to_keep = {self.current_version} if self.current_version else set()
        for version, _ in sorted_models[:keep_last_n]:
            models_to_keep.add(version)
        
        # Delete old models
        for version in list(self.models.keys()):
            if version not in models_to_keep:
                try:
                    self.delete_model(version)
                except Exception as e:
                    logger.error(f"Error deleting model {version}: {e}")
    
    def load_latest_model(self):
        """Load the most recent model as active"""
        if not self.models:
            raise ValueError("No models in registry")
        
        # Get latest model
        latest_version = max(self.models.keys(), 
                           key=lambda v: self.models[v]['created_at'])
        self.activate_model(latest_version)
    
    def _load_registry(self):
        """Load registry from file"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                self.models = data.get('models', {})
                self.current_version = data.get('current_version')
    
    def _save_registry(self):
        """Save registry to file"""
        data = {
            'models': self.models,
            'current_version': self.current_version
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)