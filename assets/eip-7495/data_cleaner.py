
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {col}")
        
        cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file_path, index=False)
        return cleaned_file_path
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    result = clean_dataset(input_file)
    if result:
        print(f"Cleaned data saved to: {result}")