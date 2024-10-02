# Importing necessary libraries to check missing values and outliers
import pandas as pd

# Loading the dataset
file_path = r"C:\Users\Hamid\Dokumen\GitHub\ProstCheck\backend\csv\por_4.csv" # bisa di sesuaikan ya
data = pd.read_csv(file_path)

# Separating numeric and non-numeric columns
numeric_data = data.select_dtypes(include='number')
non_numeric_data = data.select_dtypes(exclude='number')

# Checking missing values for both numeric and non-numeric data
missing_values_numeric = numeric_data.isnull().sum()
missing_values_non_numeric = non_numeric_data.isnull().sum()

# Calculating Q1, Q3, and IQR for numeric columns to detect outliers
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1

# Detecting outliers (values outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR])
outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR)))

# Handling outliers by replacing them with the mean of the column
for column in numeric_data.columns:
    mean_value = numeric_data[column].mean()
    numeric_data.loc[outliers[column], column] = mean_value

# Combining numeric and non-numeric data back into a single DataFrame
data_cleaned = pd.concat([numeric_data, non_numeric_data], axis=1)

# Exporting the cleaned dataset to a new CSV file
output_file_path = r"C:\Users\Hamid\Downloads\por_4_cleaned.csv" # bisa di sesuaikan ya
data_cleaned.to_csv(output_file_path, index=False)

# Generating a report for missing values and outliers
report = {
    'Missing Values (Numeric Columns)': missing_values_numeric,
    'Missing Values (Non-Numeric Columns)': missing_values_non_numeric,
    'Outliers (Numeric Columns)': outliers.sum()
}

# Displaying the report
print(report)