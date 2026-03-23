import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from preprocessing import preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(BASE_DIR, "data", "paysim_small.csv")
df = pd.read_csv(data_path)

# Preprocess data using centralized function
X, y = preprocess_data(df, for_training=True)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    class_weight='balanced',  # Helps with imbalanced data
    random_state=42
)
model.fit(X, y)

# Predictions (for evaluation)
y_pred = model.predict(X)

# Metrics
accuracy = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Save model
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
joblib.dump(model, model_path)

print("\nModel trained and saved!")