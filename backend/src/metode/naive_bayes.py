import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset yang dinormalisasi untuk model
df_normalized = pd.read_csv('csv_clean_normalisasi/prostateCancerCleanNormalisasi.csv') 
print(df_normalized.head())

# Dataset yang belum dinormalisasi untuk mengambil nilai maksimum
df_cleaned = pd.read_csv('csv_clean_normalisasi/prostateCancerClean.csv') 
print(df_cleaned.head())

# Ambil nilai maksimum dari kolom numerik
max_values = df_cleaned.drop(columns=['id', 'diagnosis_result']).max()

# Bagi X dan y label dari dataset yang dinormalisasi
X = df_normalized.drop(columns=['id', 'diagnosis_result'])
y = df_normalized['diagnosis_result']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Memprediksi Kelas pada Data Uji
y_pred = model.predict(X_test)

# Evaluasi Model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) != 0 else 0
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Menampilkan Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prediksi Negatif', 'Prediksi Positif'], 
            yticklabels=['Aktual Negatif', 'Aktual Positif'])
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.title('Confusion Matrix')
plt.show()

# Menampilkan Hasil Evaluasi
print(f'Akurasi: {accuracy * 100:.2f}%')
print(f'Presisi: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Fungsi untuk menerima input pengguna dan memprediksi hasil diagnosa
def user_input_prediction():
    print("\nMasukkan data pasien untuk melakukan diagnosa:")
    
    # Mengambil input dari pengguna
    radius = float(input("Masukkan nilai radius: "))
    texture = float(input("Masukkan nilai texture: "))
    perimeter = float(input("Masukkan nilai perimeter: "))
    area = float(input("Masukkan nilai area: "))
    smoothness = float(input("Masukkan nilai smoothness: "))
    compactness = float(input("Masukkan nilai compactness: "))
    symmetry = float(input("Masukkan nilai symmetry: "))
    fractal_dimension = float(input("Masukkan nilai fractal_dimension: "))
    
    # Normalisasi input pengguna dengan nilai maksimum yang telah disimpan
    user_data = pd.DataFrame({
        'radius': [radius / max_values['radius']],
        'texture': [texture / max_values['texture']],
        'perimeter': [perimeter / max_values['perimeter']],
        'area': [area / max_values['area']],
        'smoothness': [smoothness / max_values['smoothness']],
        'compactness': [compactness / max_values['compactness']],
        'symmetry': [symmetry / max_values['symmetry']],
        'fractal_dimension': [fractal_dimension / max_values['fractal_dimension']]
    })
    
    print(user_data)
    # Melakukan prediksi berdasarkan input pengguna
    prediction = model.predict(user_data)
    
    # Memberikan hasil diagnosis
    if prediction[0] == 'M':
        print("\nHasil Diagnosa: Positif (Kanker)")
        print("Prediksi: M", prediction[0])
    else:
        print("\nHasil Diagnosa: Negatif (Tidak ada Kanker)")
        print("Prediksi: B", prediction[0])

# Memanggil fungsi input prediksi pengguna
user_input_prediction()
