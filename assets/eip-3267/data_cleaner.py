
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

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[col] = data_clean[col].fillna(fill_value)
    
    return data_clean

def validate_data_types(data, schema):
    """
    Validate data types according to schema
    """
    validation_errors = []
    
    for column, expected_type in schema.items():
        if column not in data.columns:
            validation_errors.append(f"Missing column: {column}")
            continue
            
        actual_type = str(data[column].dtype)
        
        if expected_type == 'numeric':
            if not np.issubdtype(data[column].dtype, np.number):
                validation_errors.append(f"Column '{column}' should be numeric, got {actual_type}")
        elif expected_type == 'categorical':
            if not data[column].dtype == 'object':
                validation_errors.append(f"Column '{column}' should be categorical, got {actual_type}")
        elif expected_type == 'datetime':
            if not np.issubdtype(data[column].dtype, np.datetime64):
                validation_errors.append(f"Column '{column}' should be datetime, got {actual_type}")
    
    return len(validation_errors) == 0, validation_errors

def create_cleaning_pipeline(data, cleaning_steps):
    """
    Apply multiple cleaning steps sequentially
    """
    cleaned_data = data.copy()
    cleaning_report = {}
    
    for step in cleaning_steps:
        step_name = step.get('name', 'unnamed_step')
        step_func = step.get('function')
        step_args = step.get('args', {})
        
        try:
            if step_func == 'remove_outliers':
                result = remove_outliers_iqr(cleaned_data, **step_args)
                cleaned_data, removed = result
                cleaning_report[step_name] = f"Removed {removed} outliers"
            elif step_func == 'normalize':
                column = step_args.get('column')
                if column:
                    cleaned_data[column] = normalize_minmax(cleaned_data, column)
                    cleaning_report[step_name] = f"Normalized column: {column}"
            elif step_func == 'standardize':
                column = step_args.get('column')
                if column:
                    cleaned_data[column] = standardize_zscore(cleaned_data, column)
                    cleaning_report[step_name] = f"Standardized column: {column}"
            elif step_func == 'handle_missing':
                cleaned_data = handle_missing_values(cleaned_data, **step_args)
                cleaning_report[step_name] = "Handled missing values"
            else:
                cleaning_report[step_name] = f"Unknown function: {step_func}"
        except Exception as e:
            cleaning_report[step_name] = f"Error: {str(e)}"
    
    return cleaned_data, cleaning_report