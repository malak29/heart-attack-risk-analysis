from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class PatientData(BaseModel):
    """Schema for individual patient data"""
    
    # Demographics
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    sex: str = Field(..., regex='^(Male|Female)$', description="Patient sex")
    
    # Vital Signs
    heart_rate: float = Field(..., ge=30, le=220, description="Heart rate in bpm")
    blood_pressure: str = Field(..., regex=r'^\d{2,3}/\d{2,3}$', 
                                description="Blood pressure (systolic/diastolic)")
    
    # Medical Measurements
    cholesterol: float = Field(..., ge=50, le=500, description="Total cholesterol mg/dL")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    triglycerides: float = Field(..., ge=0, le=1000, description="Triglycerides mg/dL")
    
    # Lifestyle Factors
    exercise_hours_per_week: float = Field(..., ge=0, le=168, 
                                          description="Weekly exercise hours")
    sedentary_hours_per_day: float = Field(..., ge=0, le=24, 
                                          description="Daily sedentary hours")
    physical_activity_days_per_week: int = Field(..., ge=0, le=7, 
                                                description="Days of physical activity per week")
    diet: str = Field(..., regex='^(Healthy|Average|Unhealthy)$', 
                     description="Diet quality")
    
    # Medical History
    diabetes: bool = Field(..., description="Has diabetes")
    family_history: bool = Field(..., description="Family history of heart disease")
    smoking: bool = Field(..., description="Current smoker")
    obesity: bool = Field(..., description="Clinical obesity")
    alcohol_consumption: bool = Field(..., description="Regular alcohol consumption")
    previous_heart_problems: bool = Field(..., description="Previous heart issues")
    medication_use: bool = Field(..., description="On heart medication")
    
    # Location
    country: str = Field(..., min_length=2, max_length=100, description="Country")
    continent: str = Field(..., min_length=2, max_length=50, description="Continent")
    hemisphere: str = Field(..., regex='^(Northern|Southern) Hemisphere$', 
                          description="Hemisphere")
    
    @validator('blood_pressure')
    def validate_blood_pressure(cls, v):
        """Validate blood pressure format and values"""
        parts = v.split('/')
        systolic = int(parts[0])
        diastolic = int(parts[1])
        
        if systolic < 70 or systolic > 250:
            raise ValueError('Systolic pressure must be between 70-250')
        if diastolic < 40 or diastolic > 150:
            raise ValueError('Diastolic pressure must be between 40-150')
        if systolic <= diastolic:
            raise ValueError('Systolic must be greater than diastolic')
        
        return v
    
    @validator('exercise_hours_per_week')
    def validate_exercise_hours(cls, v, values):
        """Validate exercise hours against physical activity days"""
        if 'physical_activity_days_per_week' in values:
            max_hours = values['physical_activity_days_per_week'] * 24
            if v > max_hours:
                raise ValueError(f'Exercise hours cannot exceed {max_hours} based on activity days')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "sex": "Male",
                "heart_rate": 75,
                "blood_pressure": "120/80",
                "cholesterol": 200,
                "bmi": 25.5,
                "triglycerides": 150,
                "exercise_hours_per_week": 3.5,
                "sedentary_hours_per_day": 8,
                "physical_activity_days_per_week": 4,
                "diet": "Average",
                "diabetes": False,
                "family_history": True,
                "smoking": False,
                "obesity": False,
                "alcohol_consumption": False,
                "previous_heart_problems": False,
                "medication_use": True,
                "country": "United States",
                "continent": "North America",
                "hemisphere": "Northern Hemisphere"
            }
        }

class PatientBatchData(BaseModel):
    """Schema for batch patient data"""
    patients: List[PatientData] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "patients": [
                    {
                        "age": 45,
                        "sex": "Male",
                        "heart_rate": 75,
                        "blood_pressure": "120/80",
                        "cholesterol": 200,
                        "bmi": 25.5,
                        "triglycerides": 150,
                        "exercise_hours_per_week": 3.5,
                        "sedentary_hours_per_day": 8,
                        "physical_activity_days_per_week": 4,
                        "diet": "Average",
                        "diabetes": False,
                        "family_history": True,
                        "smoking": False,
                        "obesity": False,
                        "alcohol_consumption": False,
                        "previous_heart_problems": False,
                        "medication_use": True,
                        "country": "United States",
                        "continent": "North America",
                        "hemisphere": "Northern Hemisphere"
                    }
                ]
            }
        }

class PatientUpdate(BaseModel):
    """Schema for updating patient data"""
    age: Optional[float] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, regex='^(Male|Female)$')
    heart_rate: Optional[float] = Field(None, ge=30, le=220)
    blood_pressure: Optional[str] = Field(None, regex=r'^\d{2,3}/\d{2,3}$')
    cholesterol: Optional[float] = Field(None, ge=50, le=500)
    bmi: Optional[float] = Field(None, ge=10, le=60)
    triglycerides: Optional[float] = Field(None, ge=0, le=1000)
    exercise_hours_per_week: Optional[float] = Field(None, ge=0, le=168)
    sedentary_hours_per_day: Optional[float] = Field(None, ge=0, le=24)
    physical_activity_days_per_week: Optional[int] = Field(None, ge=0, le=7)
    diet: Optional[str] = Field(None, regex='^(Healthy|Average|Unhealthy)$')
    diabetes: Optional[bool] = None
    family_history: Optional[bool] = None
    smoking: Optional[bool] = None
    obesity: Optional[bool] = None
    alcohol_consumption: Optional[bool] = None
    previous_heart_problems: Optional[bool] = None
    medication_use: Optional[bool] = None
    country: Optional[str] = Field(None, min_length=2, max_length=100)
    continent: Optional[str] = Field(None, min_length=2, max_length=50)
    hemisphere: Optional[str] = Field(None, regex='^(Northern|Southern) Hemisphere$')