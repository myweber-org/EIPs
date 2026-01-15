import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop').
        columns (list): Specific columns to clean. If None, clean all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            continue
        
        if df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan, inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            else:
                print(f"Warning: Strategy '{strategy}' not applicable for column '{col}'. Skipping.")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'")
    else:
        print("No data to save.")

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, strategy='mean')
    save_cleaned_data(cleaned_df, output_file)