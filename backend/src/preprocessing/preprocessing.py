import pandas as pd
import numpy as np

# Membaca data dari file CSV
data = pd.read_csv('backend/csv/Prostate_Cancer.csv')

# Menampilkan nama kolom
print("Nama Kolom:")
print(data.columns)

# Membuat DataFrame
df = pd.DataFrame(data)

# Mengecek nilai yang hilang
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Fungsi untuk mendeteksi outliers menggunakan Median dan IQR
def detect_outliers_median_iqr(column):
    median = column.median()  # Median
    q1 = column.quantile(0.25)  # Kuartil pertama
    q3 = column.quantile(0.75)  # Kuartil ketiga
    iqr = q3 - q1  # Interquartile Range
    batas_bawah = q1 - 1.5 * iqr  # Batas bawah
    batas_atas = q1 + 1.5 * iqr  # Batas atas
    return batas_bawah, batas_atas, median

# Ganti outlier dengan median hingga semua outlier dihapus
def replace_outliers_with_median(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        if column not in ['id', 'diagnosis_result']:
            batas_bawah, batas_atas, median = detect_outliers_median_iqr(df[column])
            # Ganti outlier dengan median
            df[column] = np.where((df[column] < batas_bawah) | (df[column] > batas_atas), median, df[column])
    return df 

# Memanggil fungsi untuk mengganti outliers
df_cleaned = replace_outliers_with_median(df)

# Hitung jumlah outliers setelah pembersihan
def count_outliers(df):
    outliers_count = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        if column not in ['id', 'diagnosis_result']:
            batas_bawah, batas_atas, _ = detect_outliers_median_iqr(df[column])
            outliers_count[column] = df[(df[column] < batas_bawah) | (df[column] > batas_atas)].shape[0]
    return outliers_count

# Hitung jumlah outliers setelah pembersihan
outliers_count_after = count_outliers(df_cleaned)

print('\n---------------------Jumlah Outliers per Kolom Setelah Penggantian---------------------')
for column, count in outliers_count_after.items():
    print(f'{column}: {count} outliers') 

# Menyimpan hasil pembersihan
print('\n---------------------DataFrame Setelah Mengganti Outliers dengan Median---------------------')
print(df_cleaned)

# Menyimpan DataFrame yang telah dibersihkan ke dalam file CSV
df_cleaned.to_csv('backend/csv_clean_normalisasi/prostateCancerClean.csv', index=False)
print('\nDataFrame yang telah dibersihkan telah disimpan ke "prostateCancerClean.csv".')
