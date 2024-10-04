import pandas as pd
import numpy as np

# Membaca data dari file CSV
df_cleaned = pd.read_csv('backend/csv_clean_normalisasi/prostateCancerClean.csv')  

# Menampilkan nama kolom
print("Nama Kolom:")
print(df_cleaned.columns)

# Menyimpan kolom 'id' dan 'diagnosis_result' sebelum dihapus
id_column = df_cleaned['id']
diagnosis_result_column = df_cleaned['diagnosis_result']

# Menghapus kolom 'id' dan 'diagnosis_result'
df_cleaned = df_cleaned.drop(columns=['id', 'diagnosis_result'], errors='ignore')

# Cek tipe data kolom setelah penghapusan
print("\nTipe Data Kolom Setelah Penghapusan:")
print(df_cleaned.dtypes)

# Fungsi untuk melakukan normalisasi simple feature scaling
def simple_feature_scaling(df):
    df_normalized = df.copy()  
    # Memilih kolom numerik dan menghapus kolom yang tidak ingin dinormalisasi
    numeric_columns = df_normalized.select_dtypes(include=[np.number]).columns
    for column in numeric_columns: 
        max_value = df_normalized[column].max()
        if max_value != 0:
            df_normalized[column] = df_normalized[column] / max_value  # Normalisasi dengan Simple Scaling
        else:
            print(f"Kolom {column} memiliki nilai maksimum 0, tidak bisa dinormalisasi.")
    return df_normalized

# Melakukan normalisasi
df_normalized = simple_feature_scaling(df_cleaned)

# Menggabungkan kembali kolom 'id' dan 'diagnosis_result' ke DataFrame yang dinormalisasi
df_normalized['id'] = id_column
df_normalized['diagnosis_result'] = diagnosis_result_column

# Membulatkan DataFrame ke 2 desimal
df_normalized = df_normalized.round(3)

# Menampilkan DataFrame setelah normalisasi
print('\n---------------------DataFrame Setelah Normalisasi---------------------')
print(df_normalized)

# Menyimpan DataFrame yang dinormalisasi ke file CSV baru
df_normalized.to_csv('backend/csv_clean_normalisasi/normalisasiProstateCancerClean.csv', index=False)
