import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

def predict_fraud(df):

    required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig']

    # Check missing columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return df.assign(Error=f"Missing columns: {missing}")

    # Feature Engineering
    df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']

    # Select only needed features
    X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_diff']]

    # Prediction
    probs = model.predict_proba(X)[:, 1]
    df['Probability'] = probs.round(2)

    df['Prediction'] = (df['Probability'] > 0.3).astype(int)

    # Risk classification
    def get_risk(p):
        if p < 0.4:
            return "Low"
        elif p < 0.7:
            return "Medium"
        else:
            return "High"

    df['Risk'] = df['Probability'].apply(get_risk)

    return df