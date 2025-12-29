import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val != min_val:
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned shape: {df.shape}")
        print(f"Data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)