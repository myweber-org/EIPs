
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            
            if max_val != min_val:
                normalized_df[col] = (dataframe[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness
    """
    skewed_columns = []
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_columns.append((col, skewness))
    
    return sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True)

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_df = dataframe.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df

def validate_dataframe(dataframe):
    """
    Validate dataframe structure and content
    """
    validation_report = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicate_rows': dataframe.duplicated().sum(),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object', 'category']).columns)
    }
    
    return validation_report