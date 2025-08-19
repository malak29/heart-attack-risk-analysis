from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Settings
    api_title: str = "Heart Attack Prediction API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    debug_mode: bool = False
    
    # CORS Settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Model Settings
    model_dir: Path = Path("models")
    default_model_type: str = "random_forest"
    model_confidence_threshold: float = 0.5
    auto_retrain_threshold: float = 0.1
    max_models_to_keep: int = 10
    
    # Data Settings
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    processed_dir: Path = Path("data/processed")
    max_upload_size_mb: int = 100
    allowed_file_extensions: List[str] = [".csv", ".xlsx", ".json"]
    
    # Training Settings
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    
    # Monitoring Settings
    enable_monitoring: bool = True
    log_predictions: bool = True
    drift_check_interval_hours: int = 24
    performance_check_interval_hours: int = 12
    alert_email: Optional[str] = None
    
    # Database Settings
    database_url: Optional[str] = "sqlite:///./heart_attack.db"
    redis_url: Optional[str] = None
    
    # Security Settings
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    api_key_enabled: bool = False
    api_keys: Optional[List[str]] = None
    
    # MLflow Settings
    mlflow_tracking_uri: Optional[str] = "http://localhost:5000"
    mlflow_experiment_name: str = "heart_attack_prediction"
    
    # Logging Settings
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.model_dir, self.data_dir, self.upload_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)