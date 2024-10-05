import pandas as pd

# Read the data from the specified CSV file
data = pd.read_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv_clean_normalisasi\preproc.csv")

# Separate the 'id' and 'diagnosis_result' columns
id_col = data['id']
diagnosis_result_col = data['diagnosis_result']

# Scaling the feature set (Simple Feature Scaling)
X = data.drop(columns=['id', 'diagnosis_result'])
X_scaled = X / X.max()

# Append the 'id' and 'diagnosis_result' columns back to the scaled data
X_scaled['id'] = id_col
X_scaled['diagnosis_result'] = diagnosis_result_col

# Export the scaled feature set to a new CSV file
X_scaled.to_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv\scaled_features.csv", index=False)