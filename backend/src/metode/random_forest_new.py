import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Baca file normalisasi Excel
input_file_path_normalized = r"csv_clean_normalisasi/prostateCancerCleanNormalisasi.csv" # Path file hasil normalisasi

# Load the normalized dataset from Excel
data_normalized = pd.read_excel(input_file_path_normalized)

# 2. Pisahkan fitur (X) dan target (y)
X = data_normalized.drop(columns=['diagnosis_result', 'id'])  # 'id' bisa di-drop
y = data_normalized['diagnosis_result']

# 3. Bagi dataset menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Buat dan latih model Random Forest
model_rf = RandomForestClassifier(random_state=0)
model_rf.fit(X_train, y_train)

# 5. Prediksi menggunakan data uji
y_pred = model_rf.predict(X_test)

# 6. Evaluasi hasil prediksi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Tampilkan hasil evaluasi
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)
