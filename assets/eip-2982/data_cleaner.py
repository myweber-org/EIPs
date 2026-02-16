
import pandas as pd
import numpy as np

def clean_column_names(df):
    """
    Standardize column names: lowercase, replace spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def fill_missing_with_median(df, column):
    """
    Fill missing values in a specified column with its median.
    """
    if column in df.columns:
        median_val = df[column].median()
        df[column].fillna(median_val, inplace=True)
    return df

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a column using the IQR method.
    """
    if column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    """
    if column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            df[column] = 0
    return df

def clean_dataset(df, config):
    """
    Apply a series of cleaning operations based on a configuration dictionary.
    """
    df = clean_column_names(df)

    for col in config.get('fill_median', []):
        df = fill_missing_with_median(df, col)

    for col in config.get('remove_outliers', []):
        df = remove_outliers_iqr(df, col)

    for col in config.get('normalize', []):
        df = normalize_column(df, col)

    return df
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
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
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    return stats

def main():
    # Example usage
    data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 100]}
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_summary_statistics(cleaned_df, 'values'))

if __name__ == "__main__":
    main()