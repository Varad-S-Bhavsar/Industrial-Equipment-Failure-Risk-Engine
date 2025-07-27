# src/inference.py
import joblib
import os

# Load model from models folder
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl'))
model = joblib.load(model_path)

def predict_failure(X):
    return model.predict_proba(X)[:, 1]  # Return probability of class 1

def apply_risk_level(prob):
    if prob >= 0.7:
        return 'High'
    elif prob >= 0.3:
        return 'Medium'
    else:
        return 'Low'
