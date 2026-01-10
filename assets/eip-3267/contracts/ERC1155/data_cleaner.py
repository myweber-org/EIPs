import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for filling missing values.
                        Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): Specific columns to clean. If None, clean all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if df[col].isnull().any():
            if strategy == 'mean' and np.issubdtype(df[col].dtype, np.number):
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and np.issubdtype(df[col].dtype, np.number):
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            else:
                raise ValueError(f"Invalid strategy or column type for column: {col}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, strategy='mean')
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")