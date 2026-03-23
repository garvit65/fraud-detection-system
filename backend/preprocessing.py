import pandas as pd

def preprocess_data(df, for_training=False):
    """
    Preprocess the dataset for fraud detection.

    Args:
        df (pd.DataFrame): Input dataframe
        for_training (bool): If True, prepare for training (include target, drop NA).
                             If False, prepare for prediction (features only).

    Returns:
        For training: X, y (features and target)
        For prediction: df with processed features
    """
    # Normalize column names (strip spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Rename common column variations to standard names
    rename_dict = {
        'amt': 'amount',
        'oldbalanceorg': 'oldbalanceOrg',
        'old_balance': 'oldbalanceOrg',
        'newbalanceorig': 'newbalanceOrig',
        'new_balance': 'newbalanceOrig',
        'isfraud': 'isFraud'
    }
    df = df.rename(columns=rename_dict)

    # Define required columns
    required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
    if for_training:
        required_cols.append('isFraud')

    # Check for missing required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Expected: {required_cols}")

    # Feature engineering
    df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['balance_error'] = (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']
    df['is_large_txn'] = (df['amount'] > 20000).astype(int)

    # Select relevant features
    feature_cols = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'balance_diff',
        'amount_ratio',
        'balance_error',
        'is_large_txn'
    ]

    if for_training:
        # For training: include target, drop rows with NA
        df = df[feature_cols + ['isFraud']].dropna()
        X = df[feature_cols]
        y = df['isFraud']
        return X, y
    else:
        # For prediction: keep all rows, add features, handle NA by filling or dropping
        # For now, drop NA rows to keep simple
        df = df[feature_cols].dropna()
        return df