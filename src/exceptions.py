class HeartAttackPredictionException(Exception):
    """Base exception for the application"""
    pass

class DataValidationError(HeartAttackPredictionException):
    """Raised when data validation fails"""
    pass

class ModelNotFoundError(HeartAttackPredictionException):
    """Raised when a model is not found"""
    pass

class ModelTrainingError(HeartAttackPredictionException):
    """Raised when model training fails"""
    pass

class PredictionError(HeartAttackPredictionException):
    """Raised when prediction fails"""
    pass

class FeatureEngineeringError(HeartAttackPredictionException):
    """Raised when feature engineering fails"""
    pass

class DataProcessingError(HeartAttackPredictionException):
    """Raised when data processing fails"""
    pass

class ConfigurationError(HeartAttackPredictionException):
    """Raised when configuration is invalid"""
    pass

class MonitoringError(HeartAttackPredictionException):
    """Raised when monitoring operations fail"""
    pass

class APIKeyError(HeartAttackPredictionException):
    """Raised when API key validation fails"""
    pass