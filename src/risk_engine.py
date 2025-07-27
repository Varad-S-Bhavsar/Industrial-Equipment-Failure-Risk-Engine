import pandas as pd

def apply_risk_level(predictions) -> pd.Series:
    # Convert to Series if it's not already
    predictions = pd.Series(predictions)

    return predictions.apply(lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.5 else 'Low')
