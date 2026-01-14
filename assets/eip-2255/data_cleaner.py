
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    
    return filtered_df, removed_count

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    total_removed = 0
    for col in numeric_columns:
        df, removed = remove_outliers_iqr(df, col)
        total_removed += removed
        print(f"Removed {removed} outliers from column: {col}")
    
    df.to_csv(output_file, index=False)
    print(f"Data cleaning complete. Total outliers removed: {total_removed}")
    print(f"Cleaned data saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    
    try:
        cleaned_data = clean_dataset(input_path, output_path)
        print(f"Original shape: {pd.read_csv(input_path).shape}")
        print(f"Cleaned shape: {cleaned_data.shape}")
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")