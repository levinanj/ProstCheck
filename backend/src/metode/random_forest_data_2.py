import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca dataset
df_normalized = pd.read_csv('backend/csv_clean_normalisasi/normalisasiMedianCek.csv') 
print(df_normalized.head())

# Bagi X dan y label
X = df_normalized.drop(columns=['train'])
y = df_normalized['train']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Memprediksi Kelas pada Data Uji
y_pred = model.predict(X_test)

# Evaluasi Model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1 = 2 * (precision * recall) / (precision + recall)

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
    
    # Sesuaikan input sesuai dengan fitur pada dataset Anda
    lcavol = float(input("Masukkan nilai lcavol: "))
    lweight = float(input("Masukkan nilai lweight: "))
    age = float(input("Masukkan umur: "))
    lbph = float(input("Masukkan nilai lbph: "))
    svi = int(input("Masukkan nilai svi (0 atau 1): "))
    lcp = float(input("Masukkan nilai lcp: "))
    gleason = int(input("Masukkan nilai gleason: "))
    pgg45 = int(input("Masukkan nilai pgg45: "))
    lpsa = float(input("Masukkan nilai lpsa: "))
    
    # Membuat DataFrame dari input pengguna
    user_data = pd.DataFrame({
        'lcavol': [lcavol],
        'lweight': [lweight],
        'age': [age],
        'lbph': [lbph],
        'svi': [svi],
        'lcp': [lcp],
        'gleason': [gleason],
        'pgg45': [pgg45],
        'lpsa': [lpsa]
        
    })
    
    # Melakukan prediksi berdasarkan input pengguna
    prediction = model.predict(user_data)
    
    # Memberikan hasil diagnosis
    if prediction[0] == 1:
        print("\nHasil Diagnosa: Positif (Ada indikasi kanker)")
    else:
        print("\nHasil Diagnosa: Negatif (Tidak ada indikasi kanker)")

# Memanggil fungsi input prediksi pengguna
user_input_prediction()
