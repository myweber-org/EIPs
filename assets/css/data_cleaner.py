
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary stats:", calculate_summary_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary stats:", calculate_summary_stats(cleaned_df, 'value'))
    
    normalized_df = normalize_column(cleaned_df, 'value', method='zscore')
    print("\nNormalized column added. New columns:", normalized_df.columns.tolist())
    
    return normalized_df

if __name__ == "__main__":
    result_df = example_usage()
    print("\nFirst 5 rows of processed data:")
    print(result_df.head())
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing date columns.
    """
    # Remove duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_count - len(df)
    
    # Standardize date columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Format to YYYY-MM-DD if conversion successful
            df[col] = df[col].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Could not convert column {col}: {e}")
            continue
    
    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df, removed_duplicates

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False

def main():
    # Example usage
    sample_data = {
        'order_date': ['2023-01-15', '2023-01-15', '2023-02-20', None],
        'customer_id': [101, 101, 102, 103],
        'amount': [150.50, 150.50, 200.75, None],
        'product': ['A', 'A', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df, duplicates_removed = clean_dataframe(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate required columns
    required_cols = ['order_date', 'customer_id', 'amount']
    try:
        validate_data(cleaned_df, required_cols)
        print("Data validation passed")
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Export to file
    if export_cleaned_data(cleaned_df, 'cleaned_data.csv'):
        print("Data exported successfully")

if __name__ == "__main__":
    main()