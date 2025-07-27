import pandas as pd
import numpy as np
import os

# Create folder
os.makedirs('data', exist_ok=True)

def generate_sensor_data(num_samples=1000):
    np.random.seed(42)

    data = {
        'temperature': np.random.normal(70, 5, num_samples),
        'pressure': np.random.normal(30, 3, num_samples),
        'vibration': np.random.normal(0.5, 0.1, num_samples),
        'rpm': np.random.normal(1500, 100, num_samples),
        'sound': np.random.normal(60, 5, num_samples),
        'torque': np.random.normal(50, 3, num_samples),
        'failure': np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    }

    df = pd.DataFrame(data)
    df.to_csv('data/synthetic_sensor_data.csv', index=False)
    print("✅ synthetic_sensor_data.csv created with", len(df), "rows.")

if __name__ == "__main__":
    generate_sensor_data()
