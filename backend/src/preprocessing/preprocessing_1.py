import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data dari file CSV
data = pd.read_csv('backend/csv/Prostate_Cancer.csv')

print("Nama Kolom:")
print(data.columns)

df = pd.DataFrame(data)

missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])  

def detect_outliers_iqr(df):
    outliers_count = {}  # Inisialisasi dictionary untuk menghitung outliers
    outliers = pd.DataFrame()  # Inisialisasi outliers sebagai DataFrame kosong
    for column in df.select_dtypes(include=[np.number]).columns:  
        Q1 = df[column].quantile(0.25)  
        Q3 = df[column].quantile(0.75)  
        IQR = Q3 - Q1  # Menghitung IQR
        batas_bawah = Q1 - 1.5 * IQR  
        batas_atas = Q3 + 1.5 * IQR  
        
        # Menentukan outliers
        column_outliers = df[(df[column] < batas_bawah) | (df[column] > batas_atas)]
        
        # Menyimpan jumlah outliers per kolom
        outliers_count[column] = column_outliers.shape[0]

        # Gabungkan hanya jika `column_outliers` tidak kosong
        if not column_outliers.empty:
            print(f'Outliers pada kolom "{column}":')
            outliers = pd.concat([outliers, column_outliers], axis=0)

    return outliers.drop_duplicates(), outliers_count
  

# Mendeteksi outliers
outliers, outliers_count = detect_outliers_iqr(df)

# Menampilkan jumlah outliers
for column, count in outliers_count.items():
    print(f'{column}: {count} outliers')

print('\n---------------------Outliers---------------------')
print(outliers)

# Menghapus outlier dari DataFrame asli
df_cleaned = df[~df.index.isin(outliers.index)]

print('\n---------------------DataFrame Setelah Menghapus Outliers---------------------')
print(df_cleaned)

print('\nStatistik Deskriptif dari DataFrame yang Dibersihkan:')
print(df_cleaned.describe())

while True:
    new_outliers, new_outliers_count = detect_outliers_iqr(df_cleaned)

    print('\n---------------------Outliers Setelah Pembersihan---------------------')
    if new_outliers.empty:
        print("Tidak ada outlier yang tersisa setelah pembersihan.")
        break
    else:
        print(new_outliers)

    df_cleaned = df_cleaned[~df_cleaned.index.isin(new_outliers.index)]
    print('\n---------------------DataFrame Setelah Menghapus Outliers Lagi---------------------')
    print(df_cleaned)

# Menampilkan jumlah outliers untuk DataFrame yang dibersihkan
for column, count in new_outliers_count.items():
    print(f'{column}: {count} outliers')

# Visualisasi untuk memeriksa outliers sebelum dan setelah pembersihan
# plt.figure(figsize=(15, 10))

# # Boxplot sebelum pembersihan
# plt.subplot(2, 1, 1)
# sns.boxplot(data=df.select_dtypes(include=[np.number]))
# plt.title('Boxplot Sebelum Pembersihan Outlier')

# # Boxplot setelah pembersihan
# plt.subplot(2, 1, 2)
# sns.boxplot(data=df_cleaned.select_dtypes(include=[np.number]))
# plt.title('Boxplot Setelah Pembersihan Outlier')

# plt.tight_layout()
# plt.show()

# Menyimpan DataFrame yang telah dibersihkan ke dalam file CSV
df_cleaned.to_csv('backend/csv_clean_normalisasi/prostateCancerClean.csv', index=False)
print('\nDataFrame yang telah dibersihkan telah disimpan ke "prostateCancer_clean.csv".')
