import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
data = pd.read_csv("backend/csv/Prostate_Cancer.csv")

# 1. Map the diagnosis_result to numerical values: 'M' -> 1, 'B' -> 0
data['diagnosis_result'] = data['diagnosis_result'].map({'M': 1, 'B': 0})

# 2. Handling missing values (mean) and making sure to exclude non-numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 3. Outlier detection and treatment (IQR method)
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Detect outliers
outliers = data[((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Print outliers
print("Outliers detected:")
print(outliers)

# Hitung jumlah outlier per kolom
outlier_counts = {}
for col in numeric_columns:
    outlier_counts[col] = data[(data[col] < (Q1[col] - 1.5 * IQR[col])) | (data[col] > (Q3[col] + 1.5 * IQR[col]))].shape[0]

# Tampilkan jumlah outlier per kolom
print("\nJumlah Outlier per Kolom:")
for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers")

# Replace outliers with median
for col in numeric_columns:
    median = data[col].median()
    median = median.astype(data[col].dtype)  # Cast median to the same dtype as the column
    data.loc[data[col] < (Q1[col] - 1.5 * IQR[col]), col] = median
    data.loc[data[col] > (Q3[col] + 1.5 * IQR[col]), col] = median

# 4. Scaling the feature set (Simple Feature Scaling)
X = data.drop(columns=['id', 'diagnosis_result'])
X_scaled = X / X.max()

# 5. Target variable (already encoded)
y = data['diagnosis_result']

# 6. Splitting the data for training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 7. Naive Bayes Classification
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 9. Print classification report
classification_rep = classification_report(y_test, y_pred)

# Output metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)

# Function to take user input and predict
def predict_new_data():
    print("\nEnter the values for the new data point:")
    new_data = []
    for col in X.columns:
        value = float(input(f"{col}: "))
        new_data.append(value)
    
    # Scale the new data using simple feature scaling
    new_data_scaled = [new_data / X.max().values]
    
    # Predict the result
    prediction = model.predict(new_data_scaled)
    result = 'M' if prediction[0] == 1 else 'B'
    print(f"\nThe predicted diagnosis result is: {result}")

# Call the function to predict new data
predict_new_data()
