import pandas as pd
import numpy as np
import re
from typing import Optional, List, Dict, Any

class FeatureEngineer:
    """Creates and manages feature engineering pipelines"""
    
    def __init__(self):
        self.feature_columns = []
        self.fitted = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        df = df.copy()
        
        # Create medical level features
        df = self._create_medical_levels(df)
        
        # Create binary features
        df = self._create_binary_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Create risk scores
        df = self._create_risk_scores(df)
        
        # One-hot encode categorical variables
        df = self._encode_categorical(df)
        
        return df
    
    def _create_medical_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create medical level categories"""
        
        # Heart Rate Level
        if 'heart_rate' in df.columns:
            df['heart_rate_level'] = pd.cut(
                df['heart_rate'],
                bins=[0, 60, 100, 220],
                labels=['Low', 'Normal', 'High'],
                include_lowest=True
            )
        
        # Cholesterol Level
        if 'cholesterol' in df.columns:
            df['cholesterol_level'] = pd.cut(
                df['cholesterol'],
                bins=[0, 200, 240, 500],
                labels=['Normal', 'Borderline', 'High'],
                include_lowest=True
            )
        
        # BMI Categories
        if 'bmi' in df.columns:
            df['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[0, 18.5, 25, 30, 60],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                include_lowest=True
            )
        
        # Triglycerides Level
        if 'triglycerides' in df.columns:
            df['triglycerides_level'] = pd.cut(
                df['triglycerides'],
                bins=[0, 150, 200, 500, 1000],
                labels=['Normal', 'Borderline', 'High', 'Very High'],
                include_lowest=True
            )
        
        # Blood Pressure Processing
        if 'blood_pressure' in df.columns:
            df = self._process_blood_pressure(df)
        
        return df
    
    def _process_blood_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and categorize blood pressure"""
        
        def extract_bp(bp_str):
            try:
                if pd.isna(bp_str):
                    return 120, 80  # Default normal values
                match = re.match(r'(\d+)/(\d+)', str(bp_str))
                if match:
                    return int(match.group(1)), int(match.group(2))
                return 120, 80
            except:
                return 120, 80
        
        if 'blood_pressure' in df.columns:
            bp_values = df['blood_pressure'].apply(lambda x: pd.Series(extract_bp(x)))
            df['systolic'] = bp_values[0]
            df['diastolic'] = bp_values[1]
            
            # Create BP category
            conditions = [
                (df['systolic'] < 120) & (df['diastolic'] < 80),
                (df['systolic'] < 130) & (df['diastolic'] < 80),
                (df['systolic'] < 140) | (df['diastolic'] < 90),
                (df['systolic'] >= 140) | (df['diastolic'] >= 90)
            ]
            choices = ['Normal', 'Elevated', 'Stage1', 'Stage2']
            df['bp_category'] = np.select(conditions, choices, default='Stage2')
        
        return df
    
    def _create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary indicator features"""
        
        # Sex encoding
        if 'sex' in df.columns:
            df['is_male'] = (df['sex'].str.lower() == 'male').astype(int)
        
        # Physical activity
        if 'physical_activity_days_per_week' in df.columns:
            df['has_regular_activity'] = (df['physical_activity_days_per_week'] >= 3).astype(int)
            df['is_sedentary'] = (df['physical_activity_days_per_week'] == 0).astype(int)
        
        # Exercise habits
        if 'exercise_hours_per_week' in df.columns:
            df['meets_exercise_guidelines'] = (df['exercise_hours_per_week'] >= 2.5).astype(int)
        
        # Diet quality
        if 'diet' in df.columns:
            df['is_unhealthy_diet'] = (df['diet'].str.lower() == 'unhealthy').astype(int)
            df['is_healthy_diet'] = (df['diet'].str.lower() == 'healthy').astype(int)
        
        # Age groups
        if 'age' in df.columns:
            df['is_senior'] = (df['age'] >= 65).astype(int)
            df['is_middle_aged'] = ((df['age'] >= 40) & (df['age'] < 65)).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        
        # BMI-Exercise interaction
        if 'bmi' in df.columns and 'exercise_hours_per_week' in df.columns:
            df['bmi_exercise_ratio'] = df['bmi'] / (df['exercise_hours_per_week'] + 1)
        
        # Age-Family History interaction
        if 'age' in df.columns and 'family_history' in df.columns:
            df['age_family_risk'] = df['age'] * df['family_history'].astype(int)
        
        # Cholesterol-Age interaction
        if 'cholesterol' in df.columns and 'age' in df.columns:
            df['cholesterol_age_risk'] = df['cholesterol'] * df['age'] / 100
        
        return df
    
    def _create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk scores"""
        
        # Count cardiovascular risk factors
        risk_factors = ['smoking', 'diabetes', 'obesity', 'family_history', 
                       'previous_heart_problems', 'is_unhealthy_diet']
        available_factors = [f for f in risk_factors if f in df.columns]
        
        if available_factors:
            df['risk_factor_count'] = df[available_factors].sum(axis=1)
            df['high_risk_profile'] = (df['risk_factor_count'] >= 3).astype(int)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables"""
        
        # Get categorical columns
        cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude certain columns from encoding
        exclude_cols = ['heart_attack_risk', 'blood_pressure', 'patient_id']
        cat_columns = [c for c in cat_columns if c not in exclude_cols]
        
        if cat_columns:
            df = pd.get_dummies(df, columns=cat_columns, drop_first=False, prefix_sep='_')
        
        return df