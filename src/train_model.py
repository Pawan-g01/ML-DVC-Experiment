import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/iris.csv")

# Remove the DVC version-tracking column
X = df.drop(columns=["target", "dataset_version"])
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest Model - Version 1
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Model trained successfully!")
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "../models/random_forest_v2.pkl")

print("Model saved as models/random_forest_v2.pkl")