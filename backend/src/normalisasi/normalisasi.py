import pandas as pd
import numpy as np

# Membaca data dari file Excel
df_cleaned = pd.read_csv('csv_clean_normalisasi/prostateCancer_clean.csv')  

# Menampilkan nama kolom
print("Nama Kolom:")
print(df_cleaned.columns)

# Fungsi untuk melakukan normalisasi simple feature scaling
def simple_feature_scaling(df):
    df_normalized = df.copy()  
    for column in df_normalized.select_dtypes(include=[np.number]).columns:  # Hanya untuk kolom numerik
        min_value = df_normalized[column].min()
        max_value = df_normalized[column].max()
        df_normalized[column] = (df_normalized[column] - min_value) / (max_value - min_value)  # Normalisasi
    return df_normalized

# Melakukan normalisasi
df_normalized = simple_feature_scaling(df_cleaned)

# Menampilkan DataFrame setelah normalisasi
print('\n---------------------DataFrame Setelah Normalisasi---------------------')
print(df_normalized)

# Menyimpan DataFrame yang dinormalisasi ke file Excel baru
df_normalized.to_csv('csv_clean_normalisasi/data_normalized.csv', index=False) 
