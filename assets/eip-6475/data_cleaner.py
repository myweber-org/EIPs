
import numpy as np
import pandas as pd

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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
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
        'count': df[column].count(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean multiple numeric columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 100, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 90, 10],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1100, 900]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df, ['temperature', 'humidity', 'pressure'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    for column in ['temperature', 'humidity', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, column)
        print(f"Statistics for {column}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        print()
import pandas as pd
import numpy as np

def clean_data(df, drop_duplicates=True, fill_missing='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")

    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")

    if df.isnull().sum().any():
        print("\nMissing values found:")
        print(df.isnull().sum())

        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            print("Filled numeric missing values with column mean.")
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            print("Filled numeric missing values with column median.")
        elif fill_missing == 'drop':
            df = df.dropna()
            print("Dropped rows with any missing values.")
        else:
            print(f"Unsupported fill_missing method: {fill_missing}")

    print(f"\nCleaned data shape: {df.shape}")
    return df

def validate_data(df):
    """
    Performs basic validation on the cleaned DataFrame.
    """
    validation_results = {
        'has_duplicates': df.duplicated().any(),
        'has_missing_values': df.isnull().sum().any(),
        'data_types': df.dtypes.to_dict(),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan],
        'B': [10, 20, 20, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'z', 'w']
    }
    
    df = pd.DataFrame(sample_data)
    print("=== Data Cleaning Example ===")
    
    cleaned_df = clean_data(df, drop_duplicates=True, fill_missing='mean')
    
    print("\n=== Validation Results ===")
    validation = validate_data(cleaned_df)
    for key, value in validation.items():
        print(f"{key}: {value}")