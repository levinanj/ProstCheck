# Importing necessary libraries to check missing values and outliers
import pandas as pd
from scipy import stats

data = pd.read_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv\ke1.csv") #bisa di sesuaikan dengan path yang ada di lokal

# 1. Map the diagnosis_result to numerical values: 'M' -> 1, 'B' -> 0
data['diagnosis_result'] = data['diagnosis_result'].map({'M': 1, 'B': 0})

# 2. Handling missing values (mean) and making sure to exclude non-numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 3. Outlier detection and treatment (Z-score method)
z_scores = stats.zscore(data[numeric_columns])
abs_z_scores = pd.DataFrame(abs(z_scores), columns=numeric_columns)

# Laporan outlier yang terdeteksi sebelum penanganan
outliers_report = data[(abs_z_scores > 3).any(axis=1)]

print("Outliers remaining before handling:")
print(outliers_report)

# Replace outliers with median values for columns where abs_z_score > 3
threshold = 3
for col in numeric_columns:
    median_value = data[col].median()

    # Detect outliers using Z-score threshold
    outliers = abs_z_scores[col] > threshold
    
    # Convert median value to int if the column is of integer type
    if pd.api.types.is_integer_dtype(data[col]):
        median_value = int(median_value)
    
    # Replace outliers with the median value
    data.loc[outliers, col] = median_value

# Deteksi ulang outlier setelah penggantian
z_scores_after = stats.zscore(data[numeric_columns])
abs_z_scores_after = pd.DataFrame(abs(z_scores_after), columns=numeric_columns)
outliers_remaining = (abs_z_scores_after > threshold).any(axis=1)

# Print outliers yang masih ada (seharusnya tidak ada)
print("Outliers remaining after handling:")
print(data[outliers_remaining])

output_file_path = r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv_clean_normalisasi\preproc.csv"
data.to_csv(output_file_path, index=False)