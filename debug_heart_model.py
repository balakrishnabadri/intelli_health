#!/usr/bin/env python3
"""
Debug script to check heart disease model issue
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_heart_dataset():
    """Create heart disease dataset"""
    np.random.seed(42)
    n_samples = 1000
    
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
        
        # Rule-based target
        risk_score = age * 0.02 + cp * 0.1 + (trestbps - 120) * 0.005 + exang * 0.3
        target = 1 if risk_score > 2.5 else 0
        
        heart_data.append([age, sex, cp, max(0, trestbps), max(0, chol), fbs, 
                         restecg, max(0, thalach), exang, max(0, oldpeak), slope, target])
    
    heart_df = pd.DataFrame(heart_data, columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'target'
    ])
    
    return heart_df

# Create and analyze dataset
heart_df = create_heart_dataset()
print("Heart disease dataset shape:", heart_df.shape)
print("Target distribution:")
print(heart_df['target'].value_counts())
print("Target proportions:")
print(heart_df['target'].value_counts(normalize=True))

# Train model
X_heart = heart_df.drop('target', axis=1)
y_heart = heart_df['target']
X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

scaler_heart = StandardScaler()
X_train_scaled = scaler_heart.fit_transform(X_train)
X_test_scaled = scaler_heart.transform(X_test)

model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_train_scaled, y_train)

print("\nModel classes:", model_heart.classes_)
print("Number of classes:", len(model_heart.classes_))

# Test prediction
test_features = np.array([[55, 1, 2, 140.0, 250.0, 1, 0, 150.0, 0, 1.5, 1]])
test_scaled = scaler_heart.transform(test_features)

prediction = model_heart.predict(test_scaled)[0]
probabilities = model_heart.predict_proba(test_scaled)[0]

print(f"\nTest prediction: {prediction}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Probabilities: {probabilities}")

if len(probabilities) == 2:
    print(f"Probability of class 1: {probabilities[1]}")
else:
    print("WARNING: Only one class predicted!")
    print(f"Single probability: {probabilities[0]}")