import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        if columns is None:
            columns = df.columns
        
        for col in columns:
            if col in df.columns:
                if df[col].isnull().any():
                    if strategy == 'mean' and np.issubdtype(df[col].dtype, np.number):
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == 'median' and np.issubdtype(df[col].dtype, np.number):
                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                    elif strategy == 'drop':
                        df.dropna(subset=[col], inplace=True)
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Path for output CSV file
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_missing_data(input_file, strategy='median')
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)