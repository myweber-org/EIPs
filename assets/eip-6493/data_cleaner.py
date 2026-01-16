import numpy as np
import pandas as pd
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

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_mean(df, column):
    mean_val = df[column].mean()
    df[column].fillna(mean_val, inplace=True)
    return df

def process_dataset(df, numeric_columns):
    processed_df = df.copy()
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df = remove_outliers_iqr(processed_df, col)
            processed_df = normalize_minmax(processed_df, col)
            processed_df = standardize_zscore(processed_df, col)
            processed_df = handle_missing_mean(processed_df, col)
    return processed_df

if __name__ == "__main__":
    sample_data = {'A': [1, 2, 3, 4, 5, 100],
                   'B': [10, 20, 30, 40, 50, 200],
                   'C': [0.1, 0.2, 0.3, 0.4, 0.5, 2.0]}
    df = pd.DataFrame(sample_data)
    result = process_dataset(df, ['A', 'B', 'C'])
    print(result.head())
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(dataframe, column, method='zscore'):
    """
    Normalize a column using specified method.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'zscore':
        mean = dataframe[column].mean()
        std = dataframe[column].std()
        if std == 0:
            return dataframe[column]
        normalized = (dataframe[column] - mean) / std
    
    elif method == 'minmax':
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        if max_val == min_val:
            return dataframe[column]
        normalized = (dataframe[column] - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        median = dataframe[column].median()
        iqr = stats.iqr(dataframe[column])
        if iqr == 0:
            return dataframe[column]
        normalized = (dataframe[column] - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def clean_dataset(dataframe, numeric_columns, outlier_threshold=1.5, normalize_method='zscore'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            cleaned_df[column] = normalize_column(cleaned_df, column, normalize_method)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def get_summary_statistics(dataframe, numeric_columns=None):
    """
    Get summary statistics for numeric columns.
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = {}
    for column in numeric_columns:
        if column in dataframe.columns:
            col_data = dataframe[column].dropna()
            summary[column] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'count': len(col_data),
                'missing': dataframe[column].isna().sum()
            }
    
    return summary