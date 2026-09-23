import pandas as pd
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_file = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_facility_hygiene.csv"
)

df = pd.read_csv(data_file)

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

# Same train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load models
logistic_model = joblib.load(
    os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
)

random_forest_model = joblib.load(
    os.path.join(BASE_DIR, "models", "random_forest.pkl")
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

results = []

for name, model in models.items():

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("MAE      :", mae)
    print("MSE      :", mse)
    print("RMSE     :", rmse)

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "evaluation",
        "model_comparison.csv"
    ),
    index=False
)

print("\nModel comparison saved.")