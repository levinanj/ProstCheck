import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset yang dinormalisasi untuk model
df_normalized = pd.read_csv('csv_clean_normalisasi/prostateCancerCleanNormalisasi.csv') 

# Dataset yang belum dinormalisasi untuk mengambil nilai maksimum
df_cleaned = pd.read_csv('csv_clean_normalisasi/prostateCancerClean.csv') 

# Ambil nilai maksimum dari kolom numerik
max_values = df_cleaned.drop(columns=['id', 'diagnosis_result']).max()

# Bagi X dan y label dari dataset yang dinormalisasi
X = df_normalized.drop(columns=['id', 'diagnosis_result'])
y = df_normalized['diagnosis_result']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Inisialisasi model
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'Naive Bayes': GaussianNB()
}

# Simpan akurasi dan metrik lainnya
metrics_results = {}

for model_name, model in models.items():
    # Latih model
    model.fit(X_train, y_train)
    # Prediksi kelas pada data uji
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics_results[model_name] = {
        'Akurasi': accuracy,
        'Presisi': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Model': model  # Simpan model untuk prediksi selanjutnya
    }

    # Evaluasi model
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Menampilkan Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prediksi Negatif', 'Prediksi Positif'], 
                yticklabels=['Aktual Negatif', 'Aktual Positif'])
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()
    
    # Menampilkan Hasil Evaluasi
    print(f'=== {model_name} ===')
    print(f'Akurasi: {accuracy * 100:.2f}%')
    print(f'Presisi: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Visualisasi Akurasi
plt.figure(figsize=(10, 6))
plt.bar(metrics_results.keys(), [metrics['Akurasi'] for metrics in metrics_results.values()], color=['blue', 'orange', 'green'])
plt.ylabel('Akurasi')
plt.title('Perbandingan Akurasi Model Klasifikasi')
plt.ylim(0, 1)
plt.axhline(y=max([metrics['Akurasi'] for metrics in metrics_results.values()]), color='r', linestyle='--', label='Akurasi Tertinggi')
plt.legend()
plt.show()

# Rekomendasi model dengan akurasi tertinggi
best_model_name = max(metrics_results, key=lambda x: metrics_results[x]['Akurasi'])
best_model = metrics_results[best_model_name]['Model']
print(f'\nRekomendasi Model: {best_model_name} dengan Akurasi {metrics_results[best_model_name]["Akurasi"] * 100:.2f}%')

# Fungsi untuk menerima input pengguna dan memprediksi hasil diagnosa
def user_input_prediction(model):
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
    if prediction[0] == 1:
        print("\nMalignant")
        print("prediksi: M", prediction[0])
        print("\nM")
    else:
        print("prediksi: B", prediction[0])
        print("\n B")

# Memanggil fungsi input prediksi pengguna dengan model terbaik
user_input_prediction(best_model)
