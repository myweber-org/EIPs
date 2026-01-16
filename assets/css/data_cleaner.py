
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing ratio above this threshold
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    
    # Calculate missing ratio for each column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with high missing ratio
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Fill missing values based on specified method
    for column in df.columns:
        if df[column].isnull().any():
            if fill_method == 'mean':
                fill_value = df[column].mean()
            elif fill_method == 'median':
                fill_value = df[column].median()
            elif fill_method == 'mode':
                fill_value = df[column].mode()[0]
            elif fill_method == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            df[column] = df[column].fillna(fill_value)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Dropped columns: {list(columns_to_drop)}")
    print(f"Missing values filled using: {fill_method}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    output_path (str): Path to save the cleaned CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_method='median', drop_threshold=0.3)
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")