from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training, evaluation, and hyperparameter tuning"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {
            'random_forest': RandomForestClassifier,
            'gradient_boost': GradientBoostingClassifier,
            'xgboost': xgb.XGBClassifier
        }
        
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'gradient_boost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
        hyperparameters: Optional[Dict[str, Any]] = None,
        validation_split: float = 0.2
    ) -> Tuple[Any, Dict]:
        """Train a model with given parameters"""
        logger.info(f"Training {model_type} model with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Initialize model
        model_class = self.models.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if hyperparameters:
            model = model_class(**hyperparameters, random_state=42)
        else:
            model = model_class(random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            metrics['feature_importance'] = feature_importance
        
        logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return model, metrics
    
    def train_with_hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
        validation_split: float = 0.2,
        cv_folds: int = 5
    ) -> Tuple[Any, Dict]:
        """Train model with hyperparameter tuning using GridSearchCV"""
        
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Get model and parameters
        model_class = self.models.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        param_grid = self.param_grids.get(model_type, {})
        
        # Grid search
        model = model_class(random_state=42)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate
        metrics = self._evaluate_model(best_model, X_train, y_train, X_test, y_test)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_best_score'] = grid_search.best_score_
        
        # Add feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            metrics['feature_importance'] = feature_importance
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_model, metrics
    
    def _evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model performance"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if hasattr(model, 'predict_proba') else 0,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
        metrics['cv_score_mean'] = cv_scores.mean()
        metrics['cv_score_std'] = cv_scores.std()
        
        return metrics