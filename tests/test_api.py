import pytest
from fastapi.testclient import TestClient
from app import app
import json
from datetime import datetime

client = TestClient(app)

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        "age": 45.0,
        "sex": "Male",
        "heart_rate": 75.0,
        "blood_pressure": "120/80",
        "cholesterol": 200.0,
        "bmi": 25.5,
        "triglycerides": 150.0,
        "exercise_hours_per_week": 3.5,
        "sedentary_hours_per_day": 8.0,
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

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_check(self):
        """Test basic health check"""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_detailed_health(self):
        """Test detailed health check"""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "system" in data
        assert "paths" in data
    
    def test_readiness_check(self):
        """Test readiness probe"""
        response = client.get("/api/v1/health/ready")
        # May return 503 if no model is loaded
        assert response.status_code in [200, 503]
    
    def test_liveness_check(self):
        """Test liveness probe"""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] == True

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_single_prediction_no_model(self, sample_patient_data):
        """Test prediction when no model is loaded"""
        response = client.post("/api/v1/predict/", json=sample_patient_data)
        # Should return 503 if no model is available
        assert response.status_code in [503, 500]
    
    def test_invalid_patient_data(self):
        """Test prediction with invalid data"""
        invalid_data = {
            "age": -5,  # Invalid age
            "sex": "Invalid",  # Invalid sex
            "heart_rate": 300  # Invalid heart rate
        }
        response = client.post("/api/v1/predict/", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_structure(self):
        """Test batch prediction request structure"""
        batch_request = {
            "patients": [],
            "return_features": False,
            "parallel_processing": True
        }
        response = client.post("/api/v1/predict/batch", json=batch_request)
        assert response.status_code == 422  # Empty patients list

class TestTrainingEndpoints:
    """Test training endpoints"""
    
    def test_list_training_jobs(self):
        """Test listing training jobs"""
        response = client.get("/api/v1/train/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert isinstance(data["jobs"], list)
    
    def test_invalid_training_request(self):
        """Test training with invalid request"""
        invalid_request = {
            "model_type": "invalid_model",  # Invalid model type
            "validation_split": 0.5  # Too high
        }
        response = client.post("/api/v1/train/", json=invalid_request)
        assert response.status_code == 422

class TestModelManagementEndpoints:
    """Test model management endpoints"""
    
    def test_list_models(self):
        """Test listing models"""
        response = client.get("/api/v1/models/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_current_model_not_found(self):
        """Test getting current model when none exists"""
        response = client.get("/api/v1/models/current")
        # Should return 404 if no current model
        assert response.status_code in [200, 404]
    
    def test_activate_nonexistent_model(self):
        """Test activating non-existent model"""
        response = client.post("/api/v1/models/nonexistent/activate")
        assert response.status_code == 404
    
    def test_model_comparison_invalid(self):
        """Test comparing non-existent models"""
        response = client.post("/api/v1/models/compare?version1=v1&version2=v2")
        assert response.status_code == 404

class TestDataValidation:
    """Test data validation"""
    
    def test_blood_pressure_validation(self):
        """Test blood pressure format validation"""
        from src.schemas.patient import PatientData
        
        # Valid blood pressure
        valid_bp = "120/80"
        assert PatientData.model_validate_json(
            json.dumps({"blood_pressure": valid_bp, **get_minimal_patient_data()})
        )
        
        # Invalid format
        with pytest.raises(Exception):
            PatientData.model_validate_json(
                json.dumps({"blood_pressure": "invalid", **get_minimal_patient_data()})
            )
    
    def test_age_validation(self):
        """Test age range validation"""
        from src.schemas.patient import PatientData
        
        # Invalid age (negative)
        with pytest.raises(Exception):
            PatientData(age=-1, **get_minimal_patient_data())
        
        # Invalid age (too high)
        with pytest.raises(Exception):
            PatientData(age=150, **get_minimal_patient_data())

def get_minimal_patient_data():
    """Get minimal valid patient data for testing"""
    return {
        "age": 45,
        "sex": "Male",
        "heart_rate": 75,
        "blood_pressure": "120/80",
        "cholesterol": 200,
        "bmi": 25,
        "triglycerides": 150,
        "exercise_hours_per_week": 3,
        "sedentary_hours_per_day": 8,
        "physical_activity_days_per_week": 3,
        "diet": "Average",
        "diabetes": False,
        "family_history": False,
        "smoking": False,
        "obesity": False,
        "alcohol_consumption": False,
        "previous_heart_problems": False,
        "medication_use": False,
        "country": "USA",
        "continent": "North America",
        "hemisphere": "Northern Hemisphere"
    }