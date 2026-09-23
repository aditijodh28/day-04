import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_file = os.path.join(
    BASE_DIR,
    "models",
    "random_forest.pkl"
)

model = joblib.load(model_file)

# New facility data
new_facility = pd.DataFrame({
    "cleanliness_score": [5],
    "odor_score": [7],
    "waste_level": [7],
    "complaints": [6],
    "footfall": [350],
    "hours_since_cleaning": [12]
})

# Feature engineering
new_facility["complaints_per_100_people"] = (
    new_facility["complaints"] /
    new_facility["footfall"]
) * 100

new_facility["cleaning_pressure"] = (
    new_facility["hours_since_cleaning"] *
    new_facility["footfall"]
)

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

prediction = model.predict(
    new_facility[features]
)

probability = model.predict_proba(
    new_facility[features]
)

risk = "High Risk" if prediction[0] == 1 else "Low Risk"

print("Predicted Hygiene Risk:", risk)
print("Prediction Probability:", probability[0])

# Save prediction
new_facility["predicted_hygiene_risk"] = prediction
new_facility["risk_label"] = risk

output_file = os.path.join(
    BASE_DIR,
    "predictions",
    "predictions.csv"
)

new_facility.to_csv(
    output_file,
    index=False
)

print("\nPrediction saved to:")
print(output_file)
