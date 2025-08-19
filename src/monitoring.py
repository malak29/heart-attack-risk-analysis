import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitors model performance and data drift"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.predictions_log = self.logs_dir / "predictions.jsonl"
        self.metrics_log = self.logs_dir / "metrics.jsonl"
        self.drift_log = self.logs_dir / "drift.jsonl"
        
        self.baseline_stats = {}
        self.alert_thresholds = {
            'accuracy_drop': 0.1,
            'drift_threshold': 0.05,
            'prediction_time': 1.0  # seconds
        }
    
    def log_prediction(
        self,
        input_data: Dict,
        prediction: float,
        model_version: str,
        response_time: float = 0
    ) -> str:
        """Log a single prediction"""
        
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        log_entry = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'input_data': input_data,
            'prediction': prediction,
            'response_time': response_time
        }
        
        with open(self.predictions_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return prediction_id
    
    def log_metrics(self, model_version: str, metrics: Dict):
        """Log model metrics"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'metrics': metrics
        }
        
        with open(self.metrics_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def check_data_drift(
        self,
        recent_data: pd.DataFrame,
        baseline_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Check for data drift using statistical tests"""
        
        drift_results = {}
        
        # If no baseline, use stored baseline stats
        if baseline_data is None and not self.baseline_stats:
            logger.warning("No baseline data available for drift detection")
            return {'status': 'no_baseline'}
        
        # Calculate baseline stats if provided
        if baseline_data is not None:
            self._update_baseline_stats(baseline_data)
        
        # Check drift for each numeric column
        numeric_cols = recent_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.baseline_stats:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    self.baseline_stats[col]['values'],
                    recent_data[col].values
                )
                
                drift_results[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < self.alert_thresholds['drift_threshold']
                }
        
        # Log drift results
        self._log_drift(drift_results)
        
        return drift_results
    
    def get_performance_summary(
        self,
        window_hours: int = 24
    ) -> Dict:
        """Get performance summary for recent predictions"""
        
        # Load recent predictions
        recent_predictions = self._load_recent_predictions(window_hours)
        
        if not recent_predictions:
            return {'status': 'no_data'}
        
        # Calculate statistics
        predictions = [p['prediction'] for p in recent_predictions]
        response_times = [p.get('response_time', 0) for p in recent_predictions]
        
        summary = {
            'total_predictions': len(predictions),
            'avg_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'avg_response_time': np.mean(response_times),
            'max_response_time': np.max(response_times),
            'time_window_hours': window_hours
        }
        
        return summary
    
    def _update_baseline_stats(self, data: pd.DataFrame):
        """Update baseline statistics for drift detection"""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.baseline_stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'values': data[col].values
            }
    
    def _load_recent_predictions(self, window_hours: int) -> List[Dict]:
        """Load predictions within time window"""
        
        if not self.predictions_log.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_predictions = []
        
        with open(self.predictions_log, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time:
                    recent_predictions.append(entry)
        
        return recent_predictions
    
    def _log_drift(self, drift_results: Dict):
        """Log drift detection results"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'drift_results': drift_results,
            'drift_detected': any(r.get('drift_detected', False) 
                                for r in drift_results.values())
        }
        
        with open(self.drift_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')