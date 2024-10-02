# Importing necessary libraries to check missing values and outliers
import pandas as pd

# Loading the cleaned dataset
file_path = r"C:\Users\Hamid\Dokumen\GitHub\ProstCheck\backend\csv_clean_normalisasi\por_4_cleaned.csv" # bisa di sesuaikan ya
data = pd.read_csv(file_path)

# Separating numeric and non-numeric columns
numeric_data = data.select_dtypes(include='number')
non_numeric_data = data.select_dtypes(exclude='number')

# Applying simple feature scaling (min-max normalization) to numeric columns
numeric_data_normalized = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

# Combining normalized numeric and non-numeric data back into a single DataFrame
data_normalized = pd.concat([numeric_data_normalized, non_numeric_data], axis=1)

# Exporting the normalized dataset to a new CSV file
output_file_path = r"C:\Users\Hamid\Downloads\por_4_normalized.csv"   # bisa di sesuaikan ya
data_normalized.to_csv(output_file_path, index=False)

# Displaying a message to indicate that the normalization is complete
print("Data normalization complete. The normalized dataset has been saved to:", output_file_path)