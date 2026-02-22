
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[f'{column}_normalized'] = 0.5
    else:
        df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', missing_strategy='mean'):
    """
    Main function to clean a dataset by handling outliers and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_method (str): Method for outlier removal ('iqr' or None)
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
    
    return df_clean.reset_index(drop=True)
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for handling outliers ('iqr', 'zscore', 'clip')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df_clean.columns]
    
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if missing_strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif missing_strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif missing_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
        
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean_val) / std_val)
            df_clean = df_clean[z_scores < 3]
        
        elif outlier_method == 'clip':
            lower_percentile = df_clean[col].quantile(0.01)
            upper_percentile = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower=lower_percentile, upper=upper_percentile)
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    subset (list): Columns to consider for identifying duplicates
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: Dataframe with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df_norm.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_norm.columns and pd.api.types.is_numeric_dtype(df_norm[col]):
            if method == 'minmax':
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val != min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            elif method == 'standard':
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val != 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): Columns that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    stats_dict = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': stats.skew(df[column].dropna()),
        'kurtosis': stats.kurtosis(df[column].dropna())
    }
    return stats_dict

if __name__ == "__main__":
    data_path = "sample_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    cleaned_data = clean_dataset(data_path, numeric_cols)
    print(f"Original shape: {pd.read_csv(data_path).shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    
    for col in numeric_cols:
        if col in cleaned_data.columns:
            stats_info = calculate_statistics(cleaned_data, col)
            print(f"\nStatistics for {col}:")
            for key, value in stats_info.items():
                print(f"{key}: {value:.4f}")