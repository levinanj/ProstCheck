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

# 4. Outlier detection and treatment (IQR method)
def clean_outliers_IQR(data, columns, max_iter=3):
    for _ in range(max_iter):
        for col in columns:
            # Hitung Q1 (Kuartil 1) dan Q3 (Kuartil 3)
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            
            # Hitung IQR
            IQR = Q3 - Q1
            
            # Tentukan batas bawah dan batas atas untuk outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers dengan nilai median
            median_value = data[col].median()
            
            # Cek tipe kolom dan lakukan casting sesuai kebutuhan
            if pd.api.types.is_integer_dtype(data[col]):
                # Jika tipe data kolom integer, pastikan median juga dikonversi ke integer
                median_value = int(median_value)
            
            # Ganti nilai outliers dengan median
            data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = median_value
    return data

# Bersihkan outliers secara iteratif
data_cleaned = clean_outliers_IQR(data, numeric_columns)

# 5. Laporan Outlier yang Rapi
def count_outliers_IQR(data, columns):
    outliers_count = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        count_outliers = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
        outliers_count[col] = count_outliers
    return outliers_count

def count_outliers_zscore(data, columns):
    outliers_count = {}
    for col in columns:
        z_scores = stats.zscore(data[col])
        count_outliers = len(data[(z_scores > 3) | (z_scores < -3)])
        outliers_count[col] = count_outliers
    return outliers_count

# Hitung outliers dengan IQR setelah iterasi pembersihan
outliers_count_IQR = count_outliers_IQR(data_cleaned, numeric_columns)
print("\nJumlah Outliers Setelah Iterasi Pembersihan (IQR):")
for col, count in outliers_count_IQR.items():
    print(f"Kolom '{col}': {count} outliers")

# 6. Alternatif Z-Score Outlier Detection
outliers_count_zscore = count_outliers_zscore(data_cleaned, numeric_columns)
print("\nJumlah Outliers Menggunakan Z-Score (|Z| > 3):")
for col, count in outliers_count_zscore.items():
    print(f"Kolom '{col}': {count} outliers")

# 7. Export hasil ke file csv baru setelah preprocessing
data_cleaned.to_csv(output_file_path, index=False)

print(f"\nData has been cleaned and exported to {output_file_path}")
