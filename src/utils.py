import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_patient_id(data: Dict) -> str:
    """Generate unique patient ID from input data"""
    # Create hash from key patient data
    key_fields = ['age', 'sex', 'cholesterol', 'heart_rate']
    hash_input = json.dumps({k: data.get(k) for k in key_fields}, sort_keys=True)
    
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def calculate_risk_score(probabilities: np.ndarray) -> float:
    """Calculate risk score from model probabilities"""
    # Weighted average if multiple probabilities
    if len(probabilities.shape) > 1:
        return float(np.mean(probabilities[:, 1]))
    return float(probabilities[1] if len(probabilities) > 1 else probabilities[0])

def determine_risk_level(risk_score: float) -> str:
    """Determine risk level from risk score"""
    if risk_score < 0.3:
        return "Low"
    elif risk_score < 0.6:
        return "Medium"
    elif risk_score < 0.8:
        return "High"
    else:
        return "Very High"

def generate_recommendations(
    patient_data: Dict,
    risk_level: str,
    risk_factors: Optional[List[str]] = None
) -> List[str]:
    """Generate personalized health recommendations"""
    
    recommendations = []
    
    # Risk level based recommendations
    if risk_level in ["High", "Very High"]:
        recommendations.append("⚠️ Immediate consultation with a cardiologist recommended")
        recommendations.append("📊 Regular monitoring of vital signs advised")
    elif risk_level == "Medium":
        recommendations.append("🏥 Schedule a check-up with your primary care physician")
        recommendations.append("📈 Monitor blood pressure and cholesterol levels")
    
    # Specific risk factor recommendations
    if patient_data.get('smoking'):
        recommendations.append("🚭 Enroll in a smoking cessation program")
    
    if patient_data.get('bmi', 0) > 30:
        recommendations.append("🏋️ Consult a nutritionist for weight management")
        recommendations.append("🥗 Adopt a heart-healthy Mediterranean diet")
    
    if patient_data.get('exercise_hours_per_week', 0) < 2.5:
        recommendations.append("🏃 Increase physical activity to 150+ minutes/week")
    
    if patient_data.get('diabetes'):
        recommendations.append("🩺 Maintain strict blood sugar control")
    
    if patient_data.get('cholesterol', 0) > 240:
        recommendations.append("💊 Discuss statin therapy with your doctor")
    
    if patient_data.get('family_history'):
        recommendations.append("🧬 Consider genetic counseling")
    
    # General recommendations
    if risk_level in ["Low", "Medium"]:
        recommendations.append("✅ Maintain healthy lifestyle habits")
        recommendations.append("📅 Annual health check-ups recommended")
    
    return recommendations

def validate_blood_pressure(bp_string: str) -> bool:
    """Validate blood pressure format"""
    import re
    pattern = r'^\d{2,3}/\d{2,3}$'
    return bool(re.match(pattern, bp_string))

def save_results_to_csv(
    predictions: List[Dict],
    output_path: str = "predictions_output.csv"
):
    """Save prediction results to CSV file"""
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

def load_model_metrics(model_version: str, metrics_dir: str = "logs") -> Optional[Dict]:
    """Load metrics for a specific model version"""
    metrics_file = Path(metrics_dir) / "metrics.jsonl"
    
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('model_version') == model_version:
                return entry.get('metrics')
    
    return None