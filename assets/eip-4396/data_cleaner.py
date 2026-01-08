
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing columns
    with excessive missing data.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Threshold for dropping columns (0.0 to 1.0)
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    df = pd.read_csv(filepath)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing percentage per column
    missing_percent = df.isnull().sum() / len(df)
    
    # Drop columns with missing values above threshold
    columns_to_drop = missing_percent[missing_percent > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    print(f"Dropped {len(columns_to_drop)} columns with >{drop_threshold*100}% missing values")
    
    # Fill remaining missing values based on method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_method == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    elif fill_method == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        most_frequent = df[col].mode()
        if not most_frequent.empty:
            df[col] = df[col].fillna(most_frequent.iloc[0])
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Total missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned dataframe
    output_path (str): Path for output CSV file
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
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")