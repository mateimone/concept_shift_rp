import pandas as pd
import glob

# Path to your CSV files
path = './src/hyperplanes/abrupt_data_H_'

# List all CSV files in the directory
csv_files = glob.glob(path + '*.csv')

# Define the column names
columns = ['feature1', 'feature2', 'label']

# List to hold the dataframes
df_list = []

# Read each CSV file without headers and assign the column names
for file in csv_files:
    df = pd.read_csv(file, header=None, names=columns)
    df_list.append(df)

# Combine all dataframes vertically
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame to a new CSV file (optional)
combined_df.to_csv('./src/hyperplanes/combined_abrupt_drifted_dataset.csv', index=False)
