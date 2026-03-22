import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

def predict_fraud(df):

    required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig']

    if all(col in df.columns for col in required_cols):

        # Feature Engineering (same as training)
        df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_diff']]

        # Prediction
        probs = model.predict_proba(X)[:, 1]
        df['Probability'] = probs.round(2)

        df['Prediction'] = (df['Probability'] > 0.3).astype(int)

        # Probability
        probs = model.predict_proba(X)[:, 1]
        df['Probability'] = probs.round(2)

        # Risk classification
        def get_risk(p):
            if p < 0.4:
                return "Low"
            elif p < 0.7:
                return "Medium"
            else:
                return "High"

        df['Risk'] = df['Probability'].apply(get_risk)

    else:
        return pd.DataFrame({
            "Error": ["Dataset must contain: amount, oldbalanceOrg, newbalanceOrig"]
        })

    return df