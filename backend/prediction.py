import pandas as pd
import joblib
import os
from .preprocessing import preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

def predict_fraud(df):
    """
    Predict fraud for the given dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with transaction data

    Returns:
        pd.DataFrame: Dataframe with predictions and risk levels
    """
    try:
        # Store original dataframe
        df_original = df.copy()
        
        # Preprocess data using centralized function
        df_processed = preprocess_data(df, for_training=False)

        # Get features for prediction
        X = df_processed[[
            'amount',
            'oldbalanceOrg',
            'newbalanceOrig',
            'balance_diff',
            'amount_ratio',
            'balance_error',
            'is_large_txn'
        ]]

        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]
        df_processed['Probability'] = probs.round(2)

        # Predict fraud (using lower threshold for better recall)
        df_processed['Prediction'] = (df_processed['Probability'] > 0.2).astype(int)

        # Risk classification
        def get_risk(p):
            if p < 0.4:
                return "Low"
            elif p < 0.7:
                return "Medium"
            else:
                return "High"

        df_processed['Risk'] = df_processed['Probability'].apply(get_risk)

        # Reset indices for safe concatenation
        df_original = df_original.reset_index(drop=True)
        df_processed = df_processed.reset_index(drop=True)

        # Combine original data with predictions
        result = pd.concat([df_original, df_processed[['Prediction', 'Probability', 'Risk']]], axis=1)

        return result

    except Exception as e:
        # Return original df with error column if prediction fails
        return df.assign(Error=str(e))