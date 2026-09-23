import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(
    BASE_DIR,
    "dataset",
    "cleaned_facility_hygiene.csv"
)

df = pd.read_csv(file_path)

print("Dataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Distribution:")
print(df["hygiene_risk"].value_counts())

# Histogram
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")
plt.show()

# Target distribution
sns.countplot(
    x="hygiene_risk",
    data=df
)

plt.title("Hygiene Risk Distribution")
plt.xlabel("Hygiene Risk")
plt.ylabel("Count")
plt.show()