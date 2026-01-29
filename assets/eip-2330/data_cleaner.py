import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Parameters:
    file_path (str): Path to the CSV file.
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop').
    columns (list): List of column names to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            df[col] = df[col].fillna(fill_value)
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, strategy='median')
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")