import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 1. Load the scaled features and the diagnosis result from the CSV file
data = pd.read_csv(r"C:\Users\Hamid\OneDrive\Dokumen\GitHub\ProstCheck\backend\csv\scaled_features.csv", index_col=False)
X_scaled = data.drop(columns=['diagnosis_result'])
y = data['diagnosis_result']

# 2. Splitting the data for training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 3. Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=0)
}

# 4. Train models and evaluate
model_accuracies = {}
model_predictions = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Store predictions
    model_predictions[name] = y_pred
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy  # Store accuracy for comparison
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    # Output metrics
    print(f"\n=== {name} Model Evaluation ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("\nClassification Report:\n", classification_rep)

# 5. Identify the best model based on highest accuracy
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]
print(f"\nThe model with the highest accuracy is: {best_model_name}")

# 6. Function to take user input and predict using all models
def predict_new_data():
    print("\nEnter the values for the new data point:")
    new_data = []
    for col in X_scaled.columns:
        value = float(input(f"{col}: "))
        new_data.append(value)
    
    # Since the data is already scaled, we can use it directly
    new_data_scaled = [new_data]  # Wrap in list to make it 2D

    # Predict the result using all models
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(new_data_scaled)
        result = 'M' if prediction[0] == 1 else 'B'
        predictions[name] = result
        print(f"\nPrediction by {name}: {result}")
    
    # Final prediction using the best model
    final_prediction = predictions[best_model_name]
    print(f"\n=== Final Prediction (using {best_model_name}) ===")
    print(f"The predicted diagnosis result is: {final_prediction}")

# 7. Call the function to predict new data
predict_new_data()
