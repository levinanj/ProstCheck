import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load the CSV file
file_path =  r"C:\Users\Hamid\Dokumen\GitHub\ProstCheck\backend\csv_clean_normalisasi\por_4_normalized.csv" # bisa di sesuaikan ya
data = pd.read_csv(file_path)

# 2. Use 'train' column as the target variable
y = data['train']

# 3. Select features (assuming all columns except 'train' are features)
X = data.drop(columns=['train'])

# 4. Standardize the features1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Splitting the data for training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 6. Random Forest Classification
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 8. Print classification report
classification_rep = classification_report(y_test, y_pred)

# Output metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)

# 9. Function to take user input and predict
def predict_new_data():
    print("\nEnter the values for the new data point:")
    new_data = []
    for col in X.columns:
        value = float(input(f"{col}: "))
        new_data.append(value)
    
    # Scale the new data using simple feature scaling
    new_data_scaled = scaler.transform([new_data])
    
    # Predict the result
    prediction = model.predict(new_data_scaled)
    
    # Custom prediction output: 'loro iki' for True (1) and 'owh ga loro kok' for False (0)
    result = 'loro iki' if prediction[0] == 1 else 'owh ga loro kok'
    print(f"\nThe predicted result is: {result}")

# Call the function to predict new data
predict_new_data()