
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

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
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
    Handle missing values with specified strategy.
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if strategy == 'mean':
            fill_value = data[col].mean()
        elif strategy == 'median':
            fill_value = data[col].median()
        elif strategy == 'mode':
            fill_value = data[col].mode()[0] if not data[col].mode().empty else 0
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        data_filled[col] = data[col].fillna(fill_value)
    
    return data_filled

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def clean_dataset(data, config):
    """
    Comprehensive data cleaning pipeline.
    """
    if not validate_dataframe(data):
        return None
    
    cleaned_data = data.copy()
    
    if 'missing_values' in config:
        strategy = config['missing_values'].get('strategy', 'mean')
        columns = config['missing_values'].get('columns')
        cleaned_data = handle_missing_values(cleaned_data, strategy, columns)
    
    if 'outliers' in config:
        for col in config['outliers'].get('columns', []):
            if col in cleaned_data.columns:
                multiplier = config['outliers'].get('multiplier', 1.5)
                cleaned_data = remove_outliers_iqr(cleaned_data, col, multiplier)
    
    if 'normalization' in config:
        for col in config['normalization'].get('columns', []):
            if col in cleaned_data.columns:
                method = config['normalization'].get('method', 'minmax')
                if method == 'minmax':
                    cleaned_data[f"{col}_normalized"] = normalize_minmax(cleaned_data, col)
                elif method == 'zscore':
                    cleaned_data[f"{col}_standardized"] = standardize_zscore(cleaned_data, col)
    
    return cleaned_dataimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    If subset is provided, only consider specified columns for duplicates.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    Supported strategies: 'mean', 'median', 'mode', 'constant'.
    If columns is None, apply to all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns using min-max or z-score normalization.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
        elif method == 'zscore':
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std != 0:
                df_normalized[col] = (df[col] - col_mean) / col_std
            else:
                df_normalized[col] = 0
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    return df_normalized

def filter_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Filter outliers from specified columns using IQR or z-score method.
    Returns DataFrame without outliers.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filtered = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std != 0:
                z_scores = np.abs((df[col] - col_mean) / col_std)
                mask = z_scores <= threshold
            else:
                mask = pd.Series(True, index=df.index)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        df_filtered = df_filtered[mask]
    
    return df_filtered.reset_index(drop=True)

def clean_dataframe(df, steps=None):
    """
    Apply a sequence of cleaning steps to a DataFrame.
    steps should be a list of tuples (function_name, kwargs).
    """
    if steps is None:
        steps = [
            ('remove_duplicates', {}),
            ('fill_missing_values', {'strategy': 'mean'}),
            ('filter_outliers', {'method': 'iqr'})
        ]
    
    cleaned_df = df.copy()
    
    for step_name, kwargs in steps:
        if step_name == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **kwargs)
        elif step_name == 'fill_missing_values':
            cleaned_df = fill_missing_values(cleaned_df, **kwargs)
        elif step_name == 'normalize_columns':
            cleaned_df = normalize_columns(cleaned_df, **kwargs)
        elif step_name == 'filter_outliers':
            cleaned_df = filter_outliers(cleaned_df, **kwargs)
        else:
            raise ValueError(f"Unknown step: {step_name}")
    
    return cleaned_df
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nData validation result: {is_valid}")