import pandas as pd

# 1. Baca file cleaned Excel
input_file_path = r"csv_clean_normalisasi/prostateCancerClean.csv"  # Path file hasil cleaning
output_file_path_normalized = r"csv_clean_normalisasi/prostateCancerCleanNormalisasi.csv"  # Path file hasil normalisasi

# Load the cleaned dataset from Excel
data_cleaned = pd.read_csv("csv_clean_normalisasi/prostateCancerClean.csv")

# 2. Normalisasi data (Simple Feature Scaling)
# Kita akan menormalisasi semua kolom numerik kecuali target ('diagnosis_result')
X = data_cleaned.drop(columns=['id', 'diagnosis_result'])  # Drop kolom 'id' dan 'diagnosis_result'
X_scaled = X / X.max()  # Normalisasi menggunakan Simple Feature Scaling (menggunakan nilai maksimum dari setiap kolom)

# Gabungkan kembali dengan kolom diagnosis_result (target)
data_normalized = pd.concat([data_cleaned['id'], X_scaled, data_cleaned['diagnosis_result']], axis=1)

# 3. Ekspor hasil normalisasi ke file csv
data_normalized.to_csv(output_file_path_normalized, index=False)

print(f"Data has been normalized and exported to {output_file_path_normalized}")
