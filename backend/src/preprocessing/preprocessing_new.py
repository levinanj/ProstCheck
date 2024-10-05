import pandas as pd
from scipy import stats

# 1. Baca file Excel
input_file_path = r"csv/Prostate_Cancer.csv"  # Ganti dengan path file input Anda
output_file_path = r"csv_clean_normalisasi/prostateCancerClean.csv"  # Path untuk file hasil yang akan diekspor

# Load the dataset from Excel
data = pd.read_csv(input_file_path)

# 2. Map the diagnosis_result to numerical values: 'M' -> 1, 'B' -> 0
if 'diagnosis_result' in data.columns:
    data['diagnosis_result'] = data['diagnosis_result'].map({'M': 1, 'B': 0})

# 3. Handling missing values (mean) and making sure to exclude non-numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 4. Outlier detection and treatment (Z-score method)
z_scores = stats.zscore(data[numeric_columns])
abs_z_scores = pd.DataFrame(abs(z_scores), columns=numeric_columns)

# Munculkan outlier sebelum dibersihkan
print("\nOutlier Sebelum Dibersihkan:\n")
threshold = 3
outliers_count_before = {}

for col in numeric_columns:
    outliers_before = data[abs_z_scores[col] > threshold]
    count_before = len(outliers_before)
    
    if count_before > 0:
        outliers_count_before[col] = count_before
        print(f"Outliers pada kolom '{col}' sebelum dibersihkan: {count_before} outliers\n", outliers_before[[col]])

# Replace outliers with median values for columns where abs_z_score > 3
for col in numeric_columns:
    median_value = data[col].median()

    # Detect outliers using Z-score threshold
    outliers = abs_z_scores[col] > threshold

    # Convert median value to int if the column is of integer type
    if pd.api.types.is_integer_dtype(data[col]):
        median_value = int(median_value)

    # Replace outliers with the median value
    data.loc[outliers, col] = median_value

# Deteksi kembali outliers setelah dibersihkan
z_scores_after_cleaning = stats.zscore(data[numeric_columns])
abs_z_scores_after_cleaning = pd.DataFrame(abs(z_scores_after_cleaning), columns=numeric_columns)

# Munculkan outlier setelah dibersihkan
print("\nOutlier Setelah Dibersihkan:\n")
outliers_count_after = {}

for col in numeric_columns:
    outliers_after = data[abs_z_scores_after_cleaning[col] > threshold]
    count_after = len(outliers_after)
    
    if count_after > 0:
        outliers_count_after[col] = count_after
        print(f"Outliers pada kolom '{col}' setelah dibersihkan: {count_after} outliers\n", outliers_after[[col]])

# 5. Print jumlah total outliers per kolom
print("\nJumlah Outliers Sebelum Dibersihkan per Kolom:\n", outliers_count_before)
print("\nJumlah Outliers Setelah Dibersihkan per Kolom:\n", outliers_count_after)

# 6. Export hasil ke file csv baru setelah preprocessing
data.to_csv(output_file_path, index=False)

print(f"\nData has been cleaned and exported to {output_file_path}")
