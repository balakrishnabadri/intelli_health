from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="IntelliHealth Multi-Disease Prediction System")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for models and scalers
models = {}
scalers = {}

# Define Models
class PredictionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(default="anonymous")
    disease_type: str
    input_data: Dict[str, Any]
    prediction: int
    probability: float
    risk_level: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class DiabetesPrediction(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: float = Field(..., ge=0, le=300)
    blood_pressure: float = Field(..., ge=0, le=200)
    skin_thickness: float = Field(..., ge=0, le=100)
    insulin: float = Field(..., ge=0, le=900)
    bmi: float = Field(..., ge=0, le=70)
    diabetes_pedigree: float = Field(..., ge=0, le=3)
    age: int = Field(..., ge=0, le=120)

class HeartDiseasePrediction(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)  # 0: Female, 1: Male
    chest_pain_type: int = Field(..., ge=0, le=3)  # 0-3
    resting_bp: float = Field(..., ge=0, le=250)
    cholesterol: float = Field(..., ge=0, le=600)
    fasting_bs: int = Field(..., ge=0, le=1)  # 0: <120mg/dl, 1: >120mg/dl
    resting_ecg: int = Field(..., ge=0, le=2)  # 0-2
    max_hr: float = Field(..., ge=0, le=250)
    exercise_angina: int = Field(..., ge=0, le=1)  # 0: No, 1: Yes
    oldpeak: float = Field(..., ge=0, le=10)
    st_slope: int = Field(..., ge=0, le=2)  # 0-2

class ParkinsonsPrediction(BaseModel):
    mdvp_fo: float = Field(..., ge=0, le=300)  # Average vocal fundamental frequency
    mdvp_fhi: float = Field(..., ge=0, le=600)  # Maximum vocal fundamental frequency
    mdvp_flo: float = Field(..., ge=0, le=300)  # Minimum vocal fundamental frequency
    mdvp_jitter_percent: float = Field(..., ge=0, le=10)
    mdvp_jitter_abs: float = Field(..., ge=0, le=1)
    mdvp_rap: float = Field(..., ge=0, le=1)
    mdvp_ppq: float = Field(..., ge=0, le=1)
    jitter_ddp: float = Field(..., ge=0, le=1)
    mdvp_shimmer: float = Field(..., ge=0, le=1)
    mdvp_shimmer_db: float = Field(..., ge=0, le=10)
    shimmer_apq3: float = Field(..., ge=0, le=1)
    shimmer_apq5: float = Field(..., ge=0, le=1)
    mdvp_apq: float = Field(..., ge=0, le=1)
    shimmer_dda: float = Field(..., ge=0, le=1)
    nhr: float = Field(..., ge=0, le=1)
    hnr: float = Field(..., ge=0, le=50)
    rpde: float = Field(..., ge=0, le=1)
    dfa: float = Field(..., ge=0, le=1)
    spread1: float = Field(..., ge=-10, le=0)
    spread2: float = Field(..., ge=0, le=1)
    d2: float = Field(..., ge=0, le=10)
    ppe: float = Field(..., ge=0, le=1)

def create_sample_datasets():
    """Create sample datasets for training ML models"""
    
    # Diabetes dataset (simplified PIMA Indian Diabetes dataset)
    np.random.seed(42)
    n_samples = 1000
    
    diabetes_data = []
    for i in range(n_samples):
        # Create realistic but synthetic data
        age = np.random.randint(21, 81)
        pregnancies = np.random.poisson(2) if np.random.random() > 0.3 else 0
        glucose = np.random.normal(120, 30)
        bp = np.random.normal(70, 15)
        skin = np.random.normal(20, 10)
        insulin = np.random.exponential(100)
        bmi = np.random.normal(32, 8)
        dpf = np.random.gamma(2, 0.2)
        
        # Simple rule-based target creation
        risk_score = (glucose - 100) * 0.01 + (bmi - 25) * 0.02 + age * 0.005 + dpf * 0.5
        outcome = 1 if risk_score > 0.6 else 0
        
        diabetes_data.append([pregnancies, max(0, glucose), max(0, bp), max(0, skin), 
                            max(0, insulin), max(0, bmi), max(0, dpf), age, outcome])
    
    diabetes_df = pd.DataFrame(diabetes_data, columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ])
    
    # Heart Disease dataset
    heart_data = []
    for i in range(n_samples):
        age = np.random.randint(29, 78)
        sex = np.random.choice([0, 1])
        cp = np.random.choice([0, 1, 2, 3])
        trestbps = np.random.normal(130, 20)
        chol = np.random.normal(240, 50)
        fbs = np.random.choice([0, 1], p=[0.85, 0.15])
        restecg = np.random.choice([0, 1, 2])
        thalach = np.random.normal(150, 25)
        exang = np.random.choice([0, 1])
        oldpeak = np.random.exponential(1)
        slope = np.random.choice([0, 1, 2])
        
        # Improved rule-based target to ensure both classes
        risk_score = (age - 40) * 0.03 + cp * 0.15 + (trestbps - 120) * 0.01 + exang * 0.4 + (chol - 200) * 0.002
        # Add some randomness to ensure class balance
        risk_score += np.random.normal(0, 0.5)
        target = 1 if risk_score > 1.0 else 0
        
        heart_data.append([age, sex, cp, max(0, trestbps), max(0, chol), fbs, 
                         restecg, max(0, thalach), exang, max(0, oldpeak), slope, target])
    
    heart_df = pd.DataFrame(heart_data, columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'target'
    ])
    
    # Parkinson's dataset (simplified)
    parkinsons_data = []
    for i in range(n_samples):
        # Vocal frequency features
        fo = np.random.normal(150, 50)
        fhi = fo + np.random.normal(50, 20)
        flo = fo - np.random.normal(30, 15)
        
        # Jitter and shimmer features (typically higher in Parkinson's)
        jitter_pct = np.random.exponential(0.5)
        jitter_abs = jitter_pct / 100
        rap = np.random.exponential(0.3)
        ppq = np.random.exponential(0.3)
        ddp = rap * 3
        
        shimmer = np.random.exponential(0.03)
        shimmer_db = shimmer * 20
        apq3 = np.random.exponential(0.02)
        apq5 = np.random.exponential(0.03)
        apq = np.random.exponential(0.04)
        dda = apq3 * 3
        
        nhr = np.random.exponential(0.02)
        hnr = np.random.normal(25, 5)
        rpde = np.random.beta(2, 3)
        dfa = np.random.beta(5, 3)
        spread1 = np.random.normal(-6, 2)
        spread2 = np.random.beta(2, 8)
        d2 = np.random.exponential(2)
        ppe = np.random.beta(2, 8)
        
        # Rule-based target (higher jitter/shimmer = more likely Parkinson's)
        risk_score = jitter_pct * 2 + shimmer * 50 + nhr * 10 - hnr * 0.02
        status = 1 if risk_score > 2 else 0
        
        parkinsons_data.append([fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp,
                              shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                              rpde, dfa, spread1, spread2, d2, ppe, status])
    
    parkinsons_df = pd.DataFrame(parkinsons_data, columns=[
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE', 'status'
    ])
    
    return diabetes_df, heart_df, parkinsons_df

def train_models():
    """Train ML models for all diseases"""
    global models, scalers
    
    # Create datasets
    diabetes_df, heart_df, parkinsons_df = create_sample_datasets()
    
    # Train Diabetes model
    X_diabetes = diabetes_df.drop('Outcome', axis=1)
    y_diabetes = diabetes_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
    
    scaler_diabetes = StandardScaler()
    X_train_scaled = scaler_diabetes.fit_transform(X_train)
    X_test_scaled = scaler_diabetes.transform(X_test)
    
    model_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
    model_diabetes.fit(X_train_scaled, y_train)
    
    models['diabetes'] = model_diabetes
    scalers['diabetes'] = scaler_diabetes
    
    # Train Heart Disease model
    X_heart = heart_df.drop('target', axis=1)
    y_heart = heart_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
    
    scaler_heart = StandardScaler()
    X_train_scaled = scaler_heart.fit_transform(X_train)
    X_test_scaled = scaler_heart.transform(X_test)
    
    model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
    model_heart.fit(X_train_scaled, y_train)
    
    models['heart'] = model_heart
    scalers['heart'] = scaler_heart
    
    # Train Parkinson's model
    X_parkinsons = parkinsons_df.drop('status', axis=1)
    y_parkinsons = parkinsons_df['status']
    X_train, X_test, y_train, y_test = train_test_split(X_parkinsons, y_parkinsons, test_size=0.2, random_state=42)
    
    scaler_parkinsons = StandardScaler()
    X_train_scaled = scaler_parkinsons.fit_transform(X_train)
    X_test_scaled = scaler_parkinsons.transform(X_test)
    
    model_parkinsons = RandomForestClassifier(n_estimators=100, random_state=42)
    model_parkinsons.fit(X_train_scaled, y_train)
    
    models['parkinsons'] = model_parkinsons
    scalers['parkinsons'] = scaler_parkinsons
    
    logging.info("All models trained successfully!")

def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "IntelliHealth Multi-Disease Prediction System", "version": "1.0"}

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_predictions": ["diabetes", "heart", "parkinsons"]
    }

@api_router.post("/predict/diabetes")
async def predict_diabetes(data: DiabetesPrediction):
    try:
        # Convert to array for prediction
        features = np.array([[
            data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
            data.insulin, data.bmi, data.diabetes_pedigree, data.age
        ]])
        
        # Scale features
        features_scaled = scalers['diabetes'].transform(features)
        
        # Make prediction
        prediction = models['diabetes'].predict(features_scaled)[0]
        probability = models['diabetes'].predict_proba(features_scaled)[0][1]
        
        # Create result
        result = PredictionResult(
            disease_type="diabetes",
            input_data=data.dict(),
            prediction=int(prediction),
            probability=float(probability),
            risk_level=get_risk_level(probability)
        )
        
        # Save to database
        await db.predictions.insert_one(result.dict())
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict/heart")
async def predict_heart_disease(data: HeartDiseasePrediction):
    try:
        # Convert to array for prediction
        features = np.array([[
            data.age, data.sex, data.chest_pain_type, data.resting_bp, data.cholesterol,
            data.fasting_bs, data.resting_ecg, data.max_hr, data.exercise_angina,
            data.oldpeak, data.st_slope
        ]])
        
        # Scale features
        features_scaled = scalers['heart'].transform(features)
        
        # Make prediction
        prediction = models['heart'].predict(features_scaled)[0]
        probability = models['heart'].predict_proba(features_scaled)[0][1]
        
        # Create result
        result = PredictionResult(
            disease_type="heart_disease",
            input_data=data.dict(),
            prediction=int(prediction),
            probability=float(probability),
            risk_level=get_risk_level(probability)
        )
        
        # Save to database
        await db.predictions.insert_one(result.dict())
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict/parkinsons")
async def predict_parkinsons(data: ParkinsonsPrediction):
    try:
        # Convert to array for prediction
        features = np.array([[
            data.mdvp_fo, data.mdvp_fhi, data.mdvp_flo, data.mdvp_jitter_percent,
            data.mdvp_jitter_abs, data.mdvp_rap, data.mdvp_ppq, data.jitter_ddp,
            data.mdvp_shimmer, data.mdvp_shimmer_db, data.shimmer_apq3, data.shimmer_apq5,
            data.mdvp_apq, data.shimmer_dda, data.nhr, data.hnr, data.rpde, data.dfa,
            data.spread1, data.spread2, data.d2, data.ppe
        ]])
        
        # Scale features
        features_scaled = scalers['parkinsons'].transform(features)
        
        # Make prediction
        prediction = models['parkinsons'].predict(features_scaled)[0]
        probability = models['parkinsons'].predict_proba(features_scaled)[0][1]
        
        # Create result
        result = PredictionResult(
            disease_type="parkinsons",
            input_data=data.dict(),
            prediction=int(prediction),
            probability=float(probability),
            risk_level=get_risk_level(probability)
        )
        
        # Save to database
        await db.predictions.insert_one(result.dict())
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.get("/predictions/history", response_model=List[PredictionResult])
async def get_prediction_history(user_id: str = "anonymous", limit: int = 50):
    """Get prediction history for a user"""
    try:
        predictions = await db.predictions.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit).to_list(limit)
        
        return [PredictionResult(**pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@api_router.get("/predictions/stats")
async def get_prediction_stats():
    """Get overall prediction statistics"""
    try:
        total_predictions = await db.predictions.count_documents({})
        
        # Count by disease type
        pipeline = [
            {"$group": {"_id": "$disease_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        disease_stats = await db.predictions.aggregate(pipeline).to_list(None)
        
        # Count by risk level
        pipeline = [
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        risk_stats = await db.predictions.aggregate(pipeline).to_list(None)
        
        return {
            "total_predictions": total_predictions,
            "by_disease": {stat["_id"]: stat["count"] for stat in disease_stats},
            "by_risk_level": {stat["_id"]: stat["count"] for stat in risk_stats}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Training ML models...")
    train_models()
    logger.info("IntelliHealth system ready!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()