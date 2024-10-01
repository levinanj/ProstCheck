import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data dari file CSV
data = pd.read_csv('csv/Prostate_Cancer.csv')


print("Nama Kolom:")
print(data.columns)


df = pd.DataFrame(data)


missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])  


def detect_outliers_iqr(df):
    outliers = pd.DataFrame(columns=df.columns)  
    for column in df.select_dtypes(include=[np.number]).columns:  
        Q1 = df[column].quantile(0.25)  
        Q3 = df[column].quantile(0.75)  
        IQR = Q3 - Q1  # Menghitung IQR
        batas_bawah = Q1 - 1.5 * IQR  
        batas_atas = Q3 + 1.5 * IQR  
        
        # Menentukan outliers
        column_outliers = df[(df[column] < batas_bawah) | (df[column] > batas_atas)]
        outliers = pd.concat([outliers, column_outliers])  
    
    return outliers.drop_duplicates()  # Menghapus duplikat

# Mendeteksi outliers
outliers = detect_outliers_iqr(df)


print('\n---------------------Outliers---------------------')
print(outliers)

# Menghapus outlier dari DataFrame asli
df_cleaned = df[~df.index.isin(outliers.index)]

print('\n---------------------DataFrame Setelah Menghapus Outliers---------------------')
print(df_cleaned)


print('\nStatistik Deskriptif dari DataFrame yang Dibersihkan:')
print(df_cleaned.describe())


while True:
    new_outliers = detect_outliers_iqr(df_cleaned)

  
    print('\n---------------------Outliers Setelah Pembersihan---------------------')
    if new_outliers.empty:
        print("Tidak ada outlier yang tersisa setelah pembersihan.")
        break
    else:
        print(new_outliers)

  
    df_cleaned = df_cleaned[~df_cleaned.index.isin(new_outliers.index)]
    print('\n---------------------DataFrame Setelah Menghapus Outliers Lagi---------------------')
    print(df_cleaned)

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
df_cleaned.to_csv('csv_clean_normalisasi/prostateCancer_clean.csv', index=False)
print('\nDataFrame yang telah dibersihkan telah disimpan ke "prostateCancer_clean.csv".')
