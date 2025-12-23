import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, drop_na=True, fill_strategy='mean'):
    """
    Clean a CSV file by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    drop_na (bool): If True, drop rows with missing values. If False, fill them.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero').
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Remove duplicate rows
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    
    # Handle missing values
    if drop_na:
        df.dropna(inplace=True)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if fill_strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif fill_strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif fill_strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif fill_strategy == 'zero':
                    df[col].fillna(0, inplace=True)
    
    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)
    
    if output_path:
        try:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to '{output_path}'.")
            print(f"Removed {duplicates_removed} duplicate rows.")
            return None
        except Exception as e:
            print(f"Error saving file: {e}")
            return df
    else:
        print(f"Cleaning complete. Removed {duplicates_removed} duplicate rows.")
        return df

def validate_numeric_columns(df, columns):
    """
    Validate that specified columns contain only numeric values.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    columns (list): List of column names to check.
    
    Returns:
    dict: Dictionary with validation results for each column.
    """
    results = {}
    for col in columns:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            results[col] = {
                'total_values': len(df[col]),
                'non_numeric_count': non_numeric,
                'is_valid': non_numeric == 0
            }
        else:
            results[col] = {'error': 'Column not found'}
    return results

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_count = len(df)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    outliers_removed = initial_count - len(df_filtered)
    
    print(f"Removed {outliers_removed} outliers from column '{column}'.")
    return df_filtered
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, processes all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 13, 12, 14, 100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, ['values'])
    print("\nCleaned data:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary statistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")