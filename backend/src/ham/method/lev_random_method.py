import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Assuming 'data' is already loaded and contains the 'diagnosis_result' column
# Load the scaled features and the diagnosis result from the CSV file
data = pd.read_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv\scaled_features.csv", index_col=False)
X_scaled = data.drop(columns=['diagnosis_result'])
y = data['diagnosis_result']

# 6. Splitting the data for training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 7. Bayes Classification
model = RandomForestClassifier(random_state=0)
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
    for col in X_scaled.columns:
        value = float(input(f"{col}: "))
        new_data.append(value)
    
    # Scale the new data using simple feature scaling
    new_data_scaled = [new_data / X_scaled.max().values]
    
    # Predict the result
    prediction = model.predict(new_data_scaled)
    result = 'M' if prediction[0] == 1 else 'B'
    print(f"\nThe predicted diagnosis result is: {result}")

# Call the function to predict new data
predict_new_data()