import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_file = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_facility_hygiene.csv"
)

models_dir = os.path.join(BASE_DIR, "models")

df = pd.read_csv(data_file)

# Features
features = [
    "cleanliness_score",
    "odor_score",
    "waste_level",
    "complaints",
    "footfall",
    "hours_since_cleaning",
    "complaints_per_100_people",
    "cleaning_pressure"
]

X = df[features]
y = df["hygiene_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

logistic_model.fit(X_train, y_train)

# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

# Save models
joblib.dump(
    logistic_model,
    os.path.join(models_dir, "logistic_regression.pkl")
)

joblib.dump(
    random_forest_model,
    os.path.join(models_dir, "random_forest.pkl")
)

# Save test data
test_data = X_test.copy()
test_data["actual_hygiene_risk"] = y_test

test_data.to_csv(
    os.path.join(BASE_DIR, "predictions", "test_data.csv"),
    index=False
)

print("\nModels trained successfully.")

print("Saved:")
print("logistic_regression.pkl")
print("random_forest.pkl")