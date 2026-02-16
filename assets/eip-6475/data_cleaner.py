
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and optionally save the cleaned data.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save the cleaned CSV. If None, no file is saved.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop', 'zero').
    
    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        elif missing_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        else:
            raise ValueError(f"Unknown strategy: {missing_strategy}")
        
        print(f"Cleaned data shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Validation warning: DataFrame contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 
                                output_path='cleaned_sample.csv',
                                missing_strategy='mean')
    
    if validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C']):
        print("Data validation passed")
        print(cleaned_df.head())import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values: drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    # For numeric columns, fill missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

    # Remove outliers using Z-score for numeric columns
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))
    outlier_mask = (z_scores < 3).all(axis=1)
    df = df[outlier_mask]
    outliers_removed = len(outlier_mask) - outlier_mask.sum()
    print(f"Removed {outliers_removed} rows based on Z-score outlier detection.")

    # Normalize numeric columns to range [0, 1]
    if not df[numeric_cols].empty:
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
        print("Normalized numeric columns to range [0, 1].")

    print(f"Final data shape: {df.shape}")
    return df

def save_cleaned_data(df, output_filepath):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None:
        try:
            df.to_csv(output_filepath, index=False)
            print(f"Cleaned data saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)