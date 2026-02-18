
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return statsimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame and outlier indices.
    """
    clean_df = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in clean_df.columns:
            continue
            
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        col_outliers = clean_df[(clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)].index
        outlier_indices.extend(col_outliers)
        
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    
    outlier_indices = list(set(outlier_indices))
    return clean_df, outlier_indices

def normalize_minmax(df, columns):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized values.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max != col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def standardize_zscore(df, columns):
    """
    Apply z-score standardization to specified columns.
    Returns DataFrame with standardized values.
    """
    standardized_df = df.copy()
    
    for col in columns:
        if col not in standardized_df.columns:
            continue
            
        col_mean = standardized_df[col].mean()
        col_std = standardized_df[col].std()
        
        if col_std > 0:
            standardized_df[col] = (standardized_df[col] - col_mean) / col_std
        else:
            standardized_df[col] = 0
    
    return standardized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = df.columns
    
    processed_df = df.copy()
    
    if strategy == 'drop':
        return processed_df.dropna(subset=columns)
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 0
            else:
                fill_value = 0
            
            processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    Returns validation results dictionary.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {},
        'shape': df.shape
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric.append(col)
        
        if non_numeric:
            validation_results['non_numeric_columns'] = non_numeric
            validation_results['is_valid'] = False
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    return validation_results

def create_data_summary(df):
    """
    Create comprehensive summary of DataFrame.
    Returns summary dictionary.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        summary['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': value_counts.head(5).to_dict()
        }
    
    return summary