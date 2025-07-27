import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Correct path
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/synthetic_sensor_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/rf_model.pkl")

def main():
    df = pd.read_csv(DATA_PATH)

    # ✅ Ensure these exact 6 features are present in both training and inference
    features = ['temperature', 'pressure', 'vibration', 'rpm', 'sound', 'torque']
    X = df[features]
    y = df['failure']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
