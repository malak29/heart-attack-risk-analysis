"""
Data processing module for cleaning and preprocessing heart attack prediction data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import re
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data cleaning and preprocessing operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.fitted = False
        self.feature_stats = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess raw data
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting data cleaning for {len(df)} records")
        df = df.copy()
        
        # Clean column names
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Clean text data
        df = self._clean_text_columns(df)
        
        # Fix data types
        df = self._fix_data_types(df)
        
        # Remove duplicates
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        if df.shape[0] < initial_shape:
            logger.info(f"Removed {initial_shape - df.shape[0]} duplicate records")
        
        # Validate data
        df = self._validate_data(df)
        
        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently based on column type"""
        
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.debug(f"Filled {col} with median: {median_val}")
        
        # Categorical columns: fill with mode or 'Unknown'
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns - trim whitespace and standardize format"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            df[col] = df[col].str.strip()
            
            # Standardize common values
            if col in ['sex', 'gender']:
                df[col] = df[col].str.title()
                df[col] = df[col].replace({'M': 'Male', 'F': 'Female'})
            
            if col == 'diet':
                df[col] = df[col].str.title()
                
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for columns"""
        
        # Convert boolean columns
        bool_columns = ['diabetes', 'family_history', 'smoking', 'obesity', 
                       'alcohol_consumption', 'previous_heart_problems', 
                       'medication_use', 'heart_attack_risk']
        
        for col in bool_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].map({'Yes': True, 'No': False, 
                                          'yes': True, 'no': False,
                                          '1': True, '0': False,
                                          1: True, 0: False})
                else:
                    df[col] = df[col].astype(bool)
        
        # Ensure numeric columns are float
        numeric_columns = ['age', 'cholesterol', 'heart_rate', 'bmi', 
                          'triglycerides', 'exercise_hours_per_week',
                          'sedentary_hours_per_day']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges and remove invalid records"""
        initial_len = len(df)
        
        # Age validation (0-120)
        if 'age' in df.columns:
            df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        
        # BMI validation (10-60)
        if 'bmi' in df.columns:
            df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]
        
        # Heart rate validation (30-220)
        if 'heart_rate' in df.columns:
            df = df[(df['heart_rate'] >= 30) & (df['heart_rate'] <= 220)]
        
        # Cholesterol validation (50-500)
        if 'cholesterol' in df.columns:
            df = df[(df['cholesterol'] >= 50) & (df['cholesterol'] <= 500)]
        
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} records due to invalid values")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers and encoders, then transform data"""
        df = self.clean_data(df)
        
        # Fit and transform numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'heart_attack_risk']
        
        if len(numeric_cols) > 0:
            self.scaler.fit(df[numeric_cols])
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            
        self.fitted = True
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers"""
        df = self.clean_data(df)
        
        if not self.fitted:
            logger.warning("Processor not fitted. Call fit_transform first.")
            return df
        
        # Transform numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'heart_attack_risk']
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            
        return df