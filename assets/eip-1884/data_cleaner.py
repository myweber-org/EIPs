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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
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
            'median': df[col].median()
        }
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    
    stats_df = calculate_statistics(cleaned)
    print("\nCleaned data statistics:")
    print(stats_df)
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def generate_summary(df):
    summary = {
        'original_rows': len(df),
        'cleaned_rows': len(df.dropna()),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.randint(1, 1000, 200)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.iloc[10, 0] = 500
    sample_df.iloc[20, 1] = 1000
    
    cleaned = clean_dataset(sample_df, ['A', 'B', 'C'])
    stats = generate_summary(cleaned)
    print(f"Cleaning complete. Retained {stats['cleaned_rows']} rows.")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, max.
    """
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    return stats

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 13, 12, 10, 9, 8, 150]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    if subset is None:
        return df.drop_duplicates()
    return df.drop_duplicates(subset=subset)

def fill_missing_values(df, strategy='mean', columns=None):
    """Fill missing values using specified strategy."""
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'zero':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """Normalize a column using specified method."""
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """Filter outliers from a column."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

def clean_dataframe(df, operations):
    """Apply multiple cleaning operations to DataFrame."""
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, operation.get('subset'))
        elif operation['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(
                cleaned_df, 
                operation.get('strategy', 'mean'),
                operation.get('columns')
            )
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(
                cleaned_df,
                operation['column'],
                operation.get('method', 'minmax')
            )
        elif operation['type'] == 'filter_outliers':
            cleaned_df = filter_outliers(
                cleaned_df,
                operation['column'],
                operation.get('method', 'iqr'),
                operation.get('threshold', 1.5)
            )
    
    return cleaned_df