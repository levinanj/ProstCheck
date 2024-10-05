import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
data = pd.read_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv\ke1.csv") #bisa di sesuaikan dengan path yang ada di lokal

# 1. Map the diagnosis_result to numerical values: 'M' -> 1, 'B' -> 0
data['diagnosis_result'] = data['diagnosis_result'].map({'M': 1, 'B': 0})

# 2. Handling missing values (mean) and making sure to exclude non-numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 3. Outlier detection and treatment (Z-score method)
z_scores = stats.zscore(data[numeric_columns])
abs_z_scores = pd.DataFrame(abs(z_scores), columns=numeric_columns)

# Laporan outlier yang terdeteksi sebelum penanganan
outliers_report = data[(abs_z_scores > 3).any(axis=1)]

print("Outliers remaining before handling:")
print(outliers_report)

# Replace outliers with median values for columns where abs_z_score > 3
threshold = 3
for col in numeric_columns:
    median_value = data[col].median()

    # Detect outliers using Z-score threshold
    outliers = abs_z_scores[col] > threshold
    
    # Convert median value to int if the column is of integer type
    if pd.api.types.is_integer_dtype(data[col]):
        median_value = int(median_value)
    
    # Replace outliers with the median value
    data.loc[outliers, col] = median_value

# Deteksi ulang outlier setelah penggantian
z_scores_after = stats.zscore(data[numeric_columns])
abs_z_scores_after = pd.DataFrame(abs(z_scores_after), columns=numeric_columns)
outliers_remaining = (abs_z_scores_after > threshold).any(axis=1)

# Print outliers yang masih ada (seharusnya tidak ada)
print("Outliers remaining after handling:")
print(data[outliers_remaining])

# 4. Scaling the feature set (Simple Feature Scaling)(SEUSAI DENGAN PATH YANG ADA DI LOKAL)
X = data.drop(columns=['id', 'diagnosis_result'])
X_scaled = X / X.max()

# Export the scaled feature set to a new CSV file
X_scaled.to_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv\scaled_features.csv", index=False)

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

# Call the function to predIIict new data
predict_new_data()
