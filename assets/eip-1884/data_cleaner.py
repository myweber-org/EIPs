
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str, optional): Path to save the cleaned CSV file. 
                                 If None, overwrites the input file.
    subset (list, optional): Columns to consider for identifying duplicates.
    keep (str): Which duplicate to keep. Options: 'first', 'last', False.
    
    Returns:
    int: Number of duplicates removed.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_cleaned)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaning complete. Removed {duplicates_removed} duplicate rows.")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return -1
    except Exception as e:
        print(f"Error: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, columns_to_clean):
    df = pd.read_csv(file_path)
    
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    df.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Cleaned dataset saved as 'cleaned_data.csv'")
    print(f"Original shape: {pd.read_csv(file_path).shape}, Cleaned shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000)
    })
    sample_data.to_csv('sample_dataset.csv', index=False)
    
    cleaned_df = clean_dataset('sample_dataset.csv', ['feature1', 'feature2', 'feature3'])import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'mode', 'drop', 'zero'.
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned dataframe.
    """
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")
    
    original_shape = df.shape
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['mean', 'median', 'mode']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = df[col].mean()
                elif missing_strategy == 'median':
                    fill_value = df[col].median()
                elif missing_strategy == 'mode':
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                
                df[col] = df[col].fillna(fill_value)
    elif missing_strategy == 'zero':
        df = df.fillna(0)
    
    print(f"Data cleaning completed:")
    print(f"  Original shape: {original_shape}")
    print(f"  Final shape: {df.shape}")
    print(f"  Missing values handled using: {missing_strategy} strategy")
    
    return df

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers in a column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe.
    column (str): Column name to check for outliers.
    threshold (float): IQR multiplier threshold.
    
    Returns:
    pandas.Series: Boolean series indicating outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe.
    column (str): Column name to normalize.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    pandas.DataFrame: Dataframe with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[column] = 0
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[column] = 0
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
        'D': ['a', 'b', 'c', 'd', 'e']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean', columns_to_drop=['D'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df, 'C')
    print(f"\nOutliers in column C: {outliers.sum()}")
    
    normalized_df = normalize_column(cleaned_df, 'C', method='minmax')
    print("\nNormalized column C:")
    print(normalized_df[['C']].head())