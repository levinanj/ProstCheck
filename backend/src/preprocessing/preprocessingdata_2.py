import pandas as pd
import numpy as np

# Membaca data dari file CSV
data = pd.read_csv('backend/csv/prostate.csv')

# Menampilkan nama kolom
print("Nama Kolom:")
print(data.columns)

# Membuat DataFrame
df = pd.DataFrame(data)

# Mengecek nilai yang hilang
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

missing_values_numeric = df.select_dtypes(include=[np.number]).isnull().sum()
print("\nMissing Values (Numeric Columns):")
missing_values_nonNumeric = df.select_dtypes(exclude=[np.number]).isnull().sum()

# Fungsi untuk mendeteksi dan menghitung jumlah outliers menggunakan IQR
def detect_outliers_iqr(df):
    outliers_count = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        # Memeriksa apakah kolom memiliki hanya dua nilai unik (seperti data biner)
        if df[column].nunique() <= 2:
            outliers_count[column] = 0  # Tidak menghitung outliers untuk kolom biner
            continue
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1  # Menghitung IQR
        batas_bawah = Q1 - 1.5 * IQR  # Batas bawah
        batas_atas = Q3 + 1.5 * IQR  # Batas atas
        
        # Mendapatkan baris yang memiliki nilai outlier
        column_outliers = df[(df[column] < batas_bawah) | (df[column] > batas_atas)]
        
        # Menyimpan jumlah outlier untuk kolom tersebut
        outliers_count[column] = column_outliers.shape[0]
    
    return outliers_count

# Mendeteksi dan menghitung jumlah outliers
outliers_count = detect_outliers_iqr(df)
print('\n---------------------Jumlah Outliers per Kolom---------------------')
for column, count in outliers_count.items():
    print(f'{column}: {count} outliers')

# Mengganti outliers dengan nilai mean kolom terkait
def remove_outliers_while_cek_terus(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        if df[column].nunique() > 2:  # Hanya terapkan pada kolom non-biner
            mean_value = df[column].median()

            # Membulatkan nilai mean ke integer jika kolom bertipe int64
            if df[column].dtype == 'int64':
                mean_value = int(round(mean_value))
            
            while True:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                batas_bawah = Q1 - 1.5 * IQR
                batas_atas = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < batas_bawah) | (df[column] > batas_atas)]
                if outliers.empty:
                    break  # Berhenti jika tidak ada lagi outliers
                
                # Ganti outliers dengan mean
                df.loc[(df[column] < batas_bawah) | (df[column] > batas_atas), column] = mean_value

remove_outliers_while_cek_terus(df)


outliers_count = detect_outliers_iqr(df)

print('\n---------------------Jumlah Outliers per Kolom Setelah Pembersihan---------------------')
for column, count in outliers_count.items():
    print(f'{column}: {count} outliers')

# Menyimpan hasil pembersihan
df_cleaned = pd.DataFrame(df)
print('\n---------------------DataFrame Setelah Menghapus Outliers---------------------')
print(df_cleaned)

# Menyimpan DataFrame yang telah dibersihkan ke dalam file CSV
df_cleaned.to_csv('backend/csv_clean_normalisasi/prostateMeanCek.csv', index=False)
print('\nDataFrame yang telah dibersihkan telah disimpan ke "prostateCancer_clean.csv".')
