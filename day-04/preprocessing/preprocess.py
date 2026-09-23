import pandas as pd
import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(
    BASE_DIR,
    "dataset",
    "facility_hygiene.csv"
)

output_file = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_facility_hygiene.csv"
)

# Load dataset
df = pd.read_csv(input_file)

print("Original Dataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing numerical values if any
numeric_columns = [
    "cleanliness_score",
    "odor_score",
    "waste_level",
    "complaints",
    "footfall",
    "hours_since_cleaning"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing target values
df["hygiene_risk"] = df["hygiene_risk"].fillna(0)

# Feature engineering
df["complaints_per_100_people"] = (
    df["complaints"] / df["footfall"]
) * 100

df["cleaning_pressure"] = (
    df["hours_since_cleaning"] * df["footfall"]
)

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nCleaned Dataset:")
print(df)

print("\nCleaned dataset saved to:")
print(output_file)