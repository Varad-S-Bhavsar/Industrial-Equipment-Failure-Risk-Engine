import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = ['temperature', 'pressure', 'vibration', 'rpm', 'sound', 'torque']
    
    # Check for missing columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select only required features
    df = df[expected_columns].fillna(method='ffill').fillna(method='bfill')
    return df
