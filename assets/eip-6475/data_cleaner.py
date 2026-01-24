import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing ratio above this threshold
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = missing_percentage[missing_percentage > drop_threshold * 100].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == 'zero':
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
    elif fill_strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after cleaning")
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Total missing values removed: {original_shape[0]*original_shape[1] - df.shape[0]*df.shape[1]}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    output_path (str): Path to save the cleaned CSV
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return True
    return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
    
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)