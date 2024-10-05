import pandas as pd

# Read the data from the specified CSV file
data = pd.read_csv(r"C:\Users\Hamid\Dokumen\GitHub\ProstCheck\backend\csv\preprocessing_ke1.csv")

# Scaling the feature set (Simple Feature Scaling)
X = data.drop(columns=['id', 'diagnosis_result'])
X_scaled = X / X.max()

# Export the scaled feature set to a new CSV file
X_scaled.to_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv\scaled_features.csv", index=False)