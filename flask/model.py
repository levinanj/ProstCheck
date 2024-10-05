import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dataset yang dinormalisasi untuk model
df_normalized = pd.read_csv('csv/prostateCancerCleanNormalisasi.csv') 

# Dataset yang belum dinormalisasi untuk mengambil nilai maksimum
df_cleaned = pd.read_csv('csv/prostateCancerClean.csv') 

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

# Latih model dan hitung akurasi
def train_models():
    metrics_results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics_results[model_name] = {
            'Akurasi': accuracy,
            'Presisi': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Model': model
        }
    
    # Rekomendasi model terbaik berdasarkan akurasi tertinggi
    best_model_name = max(metrics_results, key=lambda x: metrics_results[x]['Akurasi'])
    best_model = metrics_results[best_model_name]['Model']
    
    return best_model, max_values

# Fungsi prediksi berdasarkan input pengguna
def predict_diagnosis(model, input_data, max_values):
    # Normalisasi input pengguna
    normalized_data = pd.DataFrame({
        'radius': [input_data['radius'] / max_values['radius']],
        'texture': [input_data['texture'] / max_values['texture']],
        'perimeter': [input_data['perimeter'] / max_values['perimeter']],
        'area': [input_data['area'] / max_values['area']],
        'smoothness': [input_data['smoothness'] / max_values['smoothness']],
        'compactness': [input_data['compactness'] / max_values['compactness']],
        'symmetry': [input_data['symmetry'] / max_values['symmetry']],
        'fractal_dimension': [input_data['fractal_dimension'] / max_values['fractal_dimension']]
    })

    # Prediksi dengan model terbaik
    prediction = model.predict(normalized_data)
    return prediction[0]  # Kembalikan 1 untuk 'Malignant', 0 untuk 'Benign'
