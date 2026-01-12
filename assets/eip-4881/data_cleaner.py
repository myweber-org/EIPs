import numpy as np
import pandas as pd

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
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.dropna()

def calculate_statistics(df):
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    return stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    
    print("Original dataset shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Cleaned dataset shape:", cleaned.shape)
    
    stats = calculate_statistics(cleaned)
    for col, values in stats.items():
        print(f"\nStatistics for {col}:")
        for stat_name, stat_value in values.items():
            print(f"  {stat_name}: {stat_value:.4f}")
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list): List of column names to normalize.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize string columns: trim whitespace and convert to lowercase
    for col in column_names:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip()).lower()
            )
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using regex.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].astype(str).str.match(email_pattern)
    
    return df

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'name': [' John Doe ', 'Jane Smith', 'John Doe', 'ALICE WONDER'],
#         'email': ['john@example.com', 'invalid-email', 'JOHN@EXAMPLE.COM', 'alice@test.org']
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned_df = clean_dataset(df, ['name'])
#     validated_df = validate_email_column(cleaned_df, 'email')
#     print(validated_df)import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
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
    
    return filtered_df

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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of removal statistics for each column
    """
    original_shape = df.shape
    removal_stats = {}
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = cleaned_df.shape[0]
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - cleaned_df.shape[0]
            removal_stats[column] = removed_count
    
    final_shape = cleaned_df.shape
    removal_stats['total_removed'] = original_shape[0] - final_shape[0]
    removal_stats['original_rows'] = original_shape[0]
    removal_stats['final_rows'] = final_shape[0]
    
    return cleaned_df, removal_stats

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 100 + 500,
        'score': np.random.randn(100) * 50 + 100
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[95, 'value'] = 2000
    df.loc[96, 'value'] = -1000
    df.loc[97, 'score'] = 500
    df.loc[98, 'score'] = -300
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal summary statistics:")
    print("Value column:", calculate_summary_statistics(df, 'value'))
    print("Score column:", calculate_summary_statistics(df, 'score'))
    
    cleaned_df, stats = clean_dataset(df, ['value', 'score'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nRemoval statistics:", stats)
    print("\nCleaned summary statistics:")
    print("Value column:", calculate_summary_statistics(cleaned_df, 'value'))
    print("Score column:", calculate_summary_statistics(cleaned_df, 'score'))