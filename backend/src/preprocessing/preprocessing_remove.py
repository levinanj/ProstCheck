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

# Menghapus baris yang memiliki nilai null
df = df.dropna()
print(f"\nJumlah baris setelah menghapus nilai null: {df.shape[0]}")

# Fungsi untuk mendeteksi outliers menggunakan IQR
def detect_outliers_iqr(column):
    q1 = column.quantile(0.25)  # Kuartil pertama
    q3 = column.quantile(0.75)  # Kuartil ketiga
    iqr = q3 - q1  # Interquartile Range
    batas_bawah = q1 - 1.5 * iqr  # Batas bawah
    batas_atas = q3 + 1.5 * iqr  # Batas atas
    return batas_bawah, batas_atas

# Hitung dan hapus outliers hingga semua outliers terhapus
def remove_all_outliers(df):
    while True:
        outliers_found = False
        for column in df.select_dtypes(include=[np.number]).columns:
            if column not in ['id', 'diagnosis_result']:
                batas_bawah, batas_atas = detect_outliers_iqr(df[column])
                # Menyimpan kondisi untuk mendeteksi outlier
                condition = (df[column] < batas_bawah) | (df[column] > batas_atas)
                if condition.any():
                    outliers_found = True
                    df = df[~condition]  # Hapus outlier
        if not outliers_found:  # Jika tidak ada outlier ditemukan, hentikan loop
            break
    return df 

# Memanggil fungsi untuk menghapus outliers
df_cleaned = remove_all_outliers(df)

# Hitung jumlah outliers setelah pembersihan
def count_outliers(df):
    outliers_count = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        if column not in ['id', 'diagnosis_result']:
            batas_bawah, batas_atas = detect_outliers_iqr(df[column])
            outliers_count[column] = df[(df[column] < batas_bawah) | (df[column] > batas_atas)].shape[0]
    return outliers_count

# Hitung jumlah outliers setelah pembersihan
outliers_count_after = count_outliers(df_cleaned)

print('\n---------------------Jumlah Outliers per Kolom Setelah Pembersihan---------------------')
for column, count in outliers_count_after.items():
    print(f'{column}: {count} outliers')  # Seharusnya 0 outliers

# Menyimpan hasil pembersihan
print('\n---------------------DataFrame Setelah Menghapus Outliers---------------------')
print(df_cleaned)

# Menyimpan DataFrame yang telah dibersihkan ke dalam file CSV
df_cleaned.to_csv('backend/csv_clean_normalisasi/prostateCancerClean.csv', index=False)
print('\nDataFrame yang telah dibersihkan telah disimpan ke "prostateCancerClean.csv".')
