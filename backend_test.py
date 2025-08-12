#!/usr/bin/env python3
"""
Backend API Tests for IntelliHealth Multi-Disease Prediction System
Tests all ML prediction endpoints and database operations
"""

import requests
import json
import time
from datetime import datetime
import sys
import os

# Get backend URL from frontend .env file
def get_backend_url():
    try:
        with open('/app/frontend/.env', 'r') as f:
            for line in f:
                if line.startswith('REACT_APP_BACKEND_URL='):
                    base_url = line.split('=')[1].strip()
                    return f"{base_url}/api"
        return "https://intellihealth.preview.emergentagent.com/api"
    except:
        return "https://intellihealth.preview.emergentagent.com/api"

BASE_URL = get_backend_url()
print(f"Testing backend at: {BASE_URL}")

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        
    def log_pass(self, test_name):
        self.passed += 1
        print(f"✅ PASS: {test_name}")
        
    def log_fail(self, test_name, error):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ FAIL: {test_name} - {error}")
        
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*60}")
        return self.failed == 0

results = TestResults()

def test_health_endpoint():
    """Test the health check endpoint and model loading"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code != 200:
            results.log_fail("Health Check", f"Status code {response.status_code}")
            return False
            
        data = response.json()
        
        # Check required fields
        if "status" not in data or data["status"] != "healthy":
            results.log_fail("Health Check", "Status not healthy")
            return False
            
        if "models_loaded" not in data or data["models_loaded"] != 3:
            results.log_fail("Health Check", f"Expected 3 models, got {data.get('models_loaded', 0)}")
            return False
            
        if "available_predictions" not in data:
            results.log_fail("Health Check", "Missing available_predictions")
            return False
            
        expected_predictions = ["diabetes", "heart", "parkinsons"]
        available = data["available_predictions"]
        if not all(pred in available for pred in expected_predictions):
            results.log_fail("Health Check", f"Missing predictions. Expected: {expected_predictions}, Got: {available}")
            return False
            
        results.log_pass("Health Check - Models loaded and available")
        return True
        
    except requests.exceptions.RequestException as e:
        results.log_fail("Health Check", f"Request failed: {str(e)}")
        return False
    except Exception as e:
        results.log_fail("Health Check", f"Unexpected error: {str(e)}")
        return False

def test_diabetes_prediction():
    """Test diabetes prediction endpoint with realistic data"""
    try:
        # Test data based on review request - realistic values
        test_data = {
            "pregnancies": 2,
            "glucose": 140.0,
            "blood_pressure": 85.0,
            "skin_thickness": 25.0,
            "insulin": 120.0,
            "bmi": 30.0,
            "diabetes_pedigree": 0.5,
            "age": 45
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/diabetes",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code != 200:
            results.log_fail("Diabetes Prediction", f"Status code {response.status_code}: {response.text}")
            return False
            
        data = response.json()
        
        # Validate response structure
        required_fields = ["id", "disease_type", "input_data", "prediction", "probability", "risk_level", "timestamp"]
        for field in required_fields:
            if field not in data:
                results.log_fail("Diabetes Prediction", f"Missing field: {field}")
                return False
        
        # Validate data types and values
        if data["disease_type"] != "diabetes":
            results.log_fail("Diabetes Prediction", f"Wrong disease type: {data['disease_type']}")
            return False
            
        if not isinstance(data["prediction"], int) or data["prediction"] not in [0, 1]:
            results.log_fail("Diabetes Prediction", f"Invalid prediction value: {data['prediction']}")
            return False
            
        if not isinstance(data["probability"], (int, float)) or not (0 <= data["probability"] <= 1):
            results.log_fail("Diabetes Prediction", f"Invalid probability: {data['probability']}")
            return False
            
        if data["risk_level"] not in ["Low Risk", "Moderate Risk", "High Risk"]:
            results.log_fail("Diabetes Prediction", f"Invalid risk level: {data['risk_level']}")
            return False
            
        results.log_pass(f"Diabetes Prediction - Risk: {data['risk_level']}, Probability: {data['probability']:.3f}")
        return True
        
    except requests.exceptions.RequestException as e:
        results.log_fail("Diabetes Prediction", f"Request failed: {str(e)}")
        return False
    except Exception as e:
        results.log_fail("Diabetes Prediction", f"Unexpected error: {str(e)}")
        return False

def test_heart_disease_prediction():
    """Test heart disease prediction endpoint with realistic data"""
    try:
        # Test data based on review request
        test_data = {
            "age": 55,
            "sex": 1,  # Male
            "chest_pain_type": 2,
            "resting_bp": 140.0,
            "cholesterol": 250.0,
            "fasting_bs": 1,  # >120mg/dl
            "resting_ecg": 0,
            "max_hr": 150.0,
            "exercise_angina": 0,
            "oldpeak": 1.5,
            "st_slope": 1
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/heart",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code != 200:
            results.log_fail("Heart Disease Prediction", f"Status code {response.status_code}: {response.text}")
            return False
            
        data = response.json()
        
        # Validate response structure
        required_fields = ["id", "disease_type", "input_data", "prediction", "probability", "risk_level", "timestamp"]
        for field in required_fields:
            if field not in data:
                results.log_fail("Heart Disease Prediction", f"Missing field: {field}")
                return False
        
        # Validate data types and values
        if data["disease_type"] != "heart_disease":
            results.log_fail("Heart Disease Prediction", f"Wrong disease type: {data['disease_type']}")
            return False
            
        if not isinstance(data["prediction"], int) or data["prediction"] not in [0, 1]:
            results.log_fail("Heart Disease Prediction", f"Invalid prediction value: {data['prediction']}")
            return False
            
        if not isinstance(data["probability"], (int, float)) or not (0 <= data["probability"] <= 1):
            results.log_fail("Heart Disease Prediction", f"Invalid probability: {data['probability']}")
            return False
            
        if data["risk_level"] not in ["Low Risk", "Moderate Risk", "High Risk"]:
            results.log_fail("Heart Disease Prediction", f"Invalid risk level: {data['risk_level']}")
            return False
            
        results.log_pass(f"Heart Disease Prediction - Risk: {data['risk_level']}, Probability: {data['probability']:.3f}")
        return True
        
    except requests.exceptions.RequestException as e:
        results.log_fail("Heart Disease Prediction", f"Request failed: {str(e)}")
        return False
    except Exception as e:
        results.log_fail("Heart Disease Prediction", f"Unexpected error: {str(e)}")
        return False

def test_parkinsons_prediction():
    """Test Parkinson's disease prediction endpoint with realistic vocal pattern data"""
    try:
        # Test data with realistic vocal frequency patterns
        test_data = {
            "mdvp_fo": 150.0,      # Average vocal fundamental frequency
            "mdvp_fhi": 200.0,     # Maximum vocal fundamental frequency  
            "mdvp_flo": 120.0,     # Minimum vocal fundamental frequency
            "mdvp_jitter_percent": 0.5,
            "mdvp_jitter_abs": 0.005,
            "mdvp_rap": 0.3,
            "mdvp_ppq": 0.3,
            "jitter_ddp": 0.9,
            "mdvp_shimmer": 0.03,
            "mdvp_shimmer_db": 0.6,
            "shimmer_apq3": 0.02,
            "shimmer_apq5": 0.03,
            "mdvp_apq": 0.04,
            "shimmer_dda": 0.06,
            "nhr": 0.02,
            "hnr": 25.0,
            "rpde": 0.5,
            "dfa": 0.7,
            "spread1": -6.0,
            "spread2": 0.2,
            "d2": 2.0,
            "ppe": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/parkinsons",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code != 200:
            results.log_fail("Parkinson's Prediction", f"Status code {response.status_code}: {response.text}")
            return False
            
        data = response.json()
        
        # Validate response structure
        required_fields = ["id", "disease_type", "input_data", "prediction", "probability", "risk_level", "timestamp"]
        for field in required_fields:
            if field not in data:
                results.log_fail("Parkinson's Prediction", f"Missing field: {field}")
                return False
        
        # Validate data types and values
        if data["disease_type"] != "parkinsons":
            results.log_fail("Parkinson's Prediction", f"Wrong disease type: {data['disease_type']}")
            return False
            
        if not isinstance(data["prediction"], int) or data["prediction"] not in [0, 1]:
            results.log_fail("Parkinson's Prediction", f"Invalid prediction value: {data['prediction']}")
            return False
            
        if not isinstance(data["probability"], (int, float)) or not (0 <= data["probability"] <= 1):
            results.log_fail("Parkinson's Prediction", f"Invalid probability: {data['probability']}")
            return False
            
        if data["risk_level"] not in ["Low Risk", "Moderate Risk", "High Risk"]:
            results.log_fail("Parkinson's Prediction", f"Invalid risk level: {data['risk_level']}")
            return False
            
        results.log_pass(f"Parkinson's Prediction - Risk: {data['risk_level']}, Probability: {data['probability']:.3f}")
        return True
        
    except requests.exceptions.RequestException as e:
        results.log_fail("Parkinson's Prediction", f"Request failed: {str(e)}")
        return False
    except Exception as e:
        results.log_fail("Parkinson's Prediction", f"Unexpected error: {str(e)}")
        return False

def test_prediction_history():
    """Test prediction history retrieval"""
    try:
        # Wait a moment to ensure predictions are stored
        time.sleep(1)
        
        response = requests.get(f"{BASE_URL}/predictions/history", timeout=10)
        
        if response.status_code != 200:
            results.log_fail("Prediction History", f"Status code {response.status_code}: {response.text}")
            return False
            
        data = response.json()
        
        if not isinstance(data, list):
            results.log_fail("Prediction History", f"Expected list, got {type(data)}")
            return False
            
        # Should have at least some predictions from previous tests
        if len(data) == 0:
            results.log_fail("Prediction History", "No predictions found in history")
            return False
            
        # Validate first prediction structure
        if len(data) > 0:
            pred = data[0]
            required_fields = ["id", "disease_type", "prediction", "probability", "risk_level", "timestamp"]
            for field in required_fields:
                if field not in pred:
                    results.log_fail("Prediction History", f"Missing field in prediction: {field}")
                    return False
        
        results.log_pass(f"Prediction History - Retrieved {len(data)} predictions")
        return True
        
    except requests.exceptions.RequestException as e:
        results.log_fail("Prediction History", f"Request failed: {str(e)}")
        return False
    except Exception as e:
        results.log_fail("Prediction History", f"Unexpected error: {str(e)}")
        return False

def test_prediction_stats():
    """Test prediction statistics endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/predictions/stats", timeout=10)
        
        if response.status_code != 200:
            results.log_fail("Prediction Stats", f"Status code {response.status_code}: {response.text}")
            return False
            
        data = response.json()
        
        # Validate response structure
        required_fields = ["total_predictions", "by_disease", "by_risk_level"]
        for field in required_fields:
            if field not in data:
                results.log_fail("Prediction Stats", f"Missing field: {field}")
                return False
        
        if not isinstance(data["total_predictions"], int):
            results.log_fail("Prediction Stats", f"Invalid total_predictions type: {type(data['total_predictions'])}")
            return False
            
        if not isinstance(data["by_disease"], dict):
            results.log_fail("Prediction Stats", f"Invalid by_disease type: {type(data['by_disease'])}")
            return False
            
        if not isinstance(data["by_risk_level"], dict):
            results.log_fail("Prediction Stats", f"Invalid by_risk_level type: {type(data['by_risk_level'])}")
            return False
        
        results.log_pass(f"Prediction Stats - Total: {data['total_predictions']}, Diseases: {len(data['by_disease'])}")
        return True
        
    except requests.exceptions.RequestException as e:
        results.log_fail("Prediction Stats", f"Request failed: {str(e)}")
        return False
    except Exception as e:
        results.log_fail("Prediction Stats", f"Unexpected error: {str(e)}")
        return False

def test_input_validation():
    """Test input validation for prediction endpoints"""
    try:
        # Test invalid diabetes data
        invalid_data = {
            "pregnancies": -1,  # Invalid: negative
            "glucose": 500,     # Invalid: too high
            "blood_pressure": 85.0,
            "skin_thickness": 25.0,
            "insulin": 120.0,
            "bmi": 30.0,
            "diabetes_pedigree": 0.5,
            "age": 45
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/diabetes",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        # Should return 422 for validation error
        if response.status_code != 422:
            results.log_fail("Input Validation", f"Expected 422 for invalid data, got {response.status_code}")
            return False
            
        results.log_pass("Input Validation - Properly rejects invalid data")
        return True
        
    except requests.exceptions.RequestException as e:
        results.log_fail("Input Validation", f"Request failed: {str(e)}")
        return False
    except Exception as e:
        results.log_fail("Input Validation", f"Unexpected error: {str(e)}")
        return False

def run_all_tests():
    """Run all backend tests"""
    print("Starting IntelliHealth Backend API Tests...")
    print(f"Testing against: {BASE_URL}")
    print("="*60)
    
    # Test health endpoint first
    if not test_health_endpoint():
        print("❌ Health check failed - stopping tests")
        return False
    
    # Test all prediction endpoints
    test_diabetes_prediction()
    test_heart_disease_prediction() 
    test_parkinsons_prediction()
    
    # Test history and stats
    test_prediction_history()
    test_prediction_stats()
    
    # Test validation
    test_input_validation()
    
    # Print summary
    success = results.summary()
    return success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)