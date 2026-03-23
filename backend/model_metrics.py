import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .preprocessing import preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "paysim_small.csv")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

def get_model_metrics():
    """
    Load the trained model and compute evaluation metrics.
    Returns a dictionary with model performance metrics.
    """
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Load and preprocess data
        df = pd.read_csv(data_path)
        X, y = preprocess_data(df, for_training=True)
        
        # Make predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Compute metrics
        metrics = {
            'accuracy': round(accuracy_score(y, y_pred), 4),
            'precision': round(precision_score(y, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y, y_proba), 4),
            'total_samples': len(y),
            'fraud_cases': int(y.sum()),
            'normal_cases': int((1 - y).sum()),
            'fraud_percentage': round((y.sum() / len(y)) * 100, 2)
        }
        
        return metrics
    
    except Exception as e:
        return {
            'error': str(e),
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'roc_auc': 0,
            'total_samples': 0,
            'fraud_cases': 0,
            'normal_cases': 0,
            'fraud_percentage': 0
        }
