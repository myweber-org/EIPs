
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
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
    Clean numeric data by removing NaN values and converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, cleans all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=[col])
    
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print(f"Original shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned data (outliers removed):")
    print(cleaned_df)
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    stats = calculate_statistics(cleaned_df, 'values')
    print("\nStatistics for cleaned data:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask where True indicates an outlier.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(df, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns a cleaned DataFrame.
    """
    clean_df = df.copy()
    for col in columns:
        if col in clean_df.columns:
            outlier_mask = detect_outliers_iqr(clean_df, col, threshold)
            clean_df = clean_df[~outlier_mask]
    return clean_df.reset_index(drop=True)

def normalize_minmax(data, column):
    """
    Apply min-max normalization to a column.
    Returns normalized values between 0 and 1.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Apply z-score standardization to a column.
    Returns standardized values with mean=0 and std=1.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization='standard'):
    """
    Comprehensive data cleaning pipeline.
    Handles outliers and applies normalization/standardization.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = remove_outliers(df, numeric_columns, outlier_threshold)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization == 'minmax':
                cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            elif normalization == 'standard':
                cleaned_df[f'{col}_standardized'] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def summarize_cleaning(df_before, df_after, numeric_columns):
    """
    Generate a summary of the cleaning process.
    """
    summary = {
        'original_rows': len(df_before),
        'cleaned_rows': len(df_after),
        'removed_rows': len(df_before) - len(df_after),
        'removed_percentage': ((len(df_before) - len(df_after)) / len(df_before)) * 100
    }
    
    for col in numeric_columns:
        if col in df_before.columns and col in df_after.columns:
            summary[f'{col}_original_mean'] = df_before[col].mean()
            summary[f'{col}_cleaned_mean'] = df_after[col].mean()
            summary[f'{col}_original_std'] = df_before[col].std()
            summary[f'{col}_cleaned_std'] = df_after[col].std()
    
    return pd.Series(summary)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(data, numeric_columns, outlier_factor=1.5, normalization_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, col, outlier_factor)
            removal_stats[col] = removed
            
            if normalization_method == 'minmax':
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
            elif normalization_method == 'zscore':
                cleaned_data[f'{col}_normalized'] = z_score_normalize(cleaned_data, col)
            else:
                raise ValueError("Normalization method must be 'minmax' or 'zscore'")
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns, allow_nan=False):
    """
    Validate data structure and content
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_counts = data[required_columns].isna().sum()
        if nan_counts.any():
            raise ValueError(f"NaN values found in columns: {nan_counts[nan_counts > 0].to_dict()}")
    
    return True