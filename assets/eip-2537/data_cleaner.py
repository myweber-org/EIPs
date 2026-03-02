
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
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def validate_dataframe(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df
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

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': np.random.normal(100, 15, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Cleaned data shape:", cleaned_data.shape)
    
    stats = calculate_basic_stats(cleaned_data, 'values')
    print("Statistics after cleaning:", stats)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize, None for all numeric columns
    feature_range (tuple): Desired range of transformed data (default 0-1)
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max != col_min:
                df_norm[col] = (df[col] - col_min) / (col_max - col_min)
                df_norm[col] = df_norm[col] * (max_val - min_val) + min_val
            else:
                df_norm[col] = min_val
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process, None for all columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_processed = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col] = df_processed[col].fillna(df[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col] = df_processed[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_processed[col] = df_processed[col].fillna(df[col].mode()[0])
    
    return df_processed.reset_index(drop=True)

def z_score_normalize(df, columns=None, threshold=3):
    """
    Normalize data using Z-score and optionally remove extreme outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    threshold (float): Z-score threshold for outlier removal (None to keep all)
    
    Returns:
    pd.DataFrame: Z-score normalized dataframe
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_z = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_z[col] = (df[col] - mean_val) / std_val
                
                if threshold is not None:
                    mask = np.abs(df_z[col]) <= threshold
                    df_z = df_z[mask]
            else:
                df_z[col] = 0
    
    return df_z.reset_index(drop=True)

def create_cleaning_pipeline(df, steps):
    """
    Apply multiple cleaning steps in sequence.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    steps (list): List of cleaning step configurations
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    result = df.copy()
    
    for step in steps:
        step_name = step.get('name')
        step_args = step.get('args', {})
        
        if step_name == 'remove_outliers':
            result = remove_outliers_iqr(result, **step_args)
        elif step_name == 'normalize':
            result = normalize_minmax(result, **step_args)
        elif step_name == 'handle_missing':
            result = handle_missing_values(result, **step_args)
        elif step_name == 'z_score':
            result = z_score_normalize(result, **step_args)
    
    return result