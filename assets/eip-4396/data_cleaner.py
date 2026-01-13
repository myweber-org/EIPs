
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_method='iqr', fill_value=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode', 'constant').
    outlier_method (str): Method for outlier detection ('iqr' or 'zscore').
    fill_value: Value to use for constant imputation if strategy='constant'.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    # Handle missing values
    if strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    elif strategy == 'constant':
        if fill_value is not None:
            df_clean = df_clean.fillna(fill_value)
        else:
            raise ValueError("fill_value must be provided for constant imputation")
    else:
        raise ValueError("Invalid imputation strategy")

    # Handle outliers for numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
                                      df_clean[col].median(), df_clean[col])
        elif outlier_method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean[col] = np.where(z_scores > 3, df_clean[col].median(), df_clean[col])
        else:
            raise ValueError("Invalid outlier detection method")

    return df_clean

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, np.nan, 4, 100],
#         'B': [5, np.nan, 7, 8, 9],
#         'C': ['x', 'y', 'z', 'x', 'y']
#     })
#     cleaned = clean_dataset(sample_data, strategy='median', outlier_method='iqr')
#     print("Original DataFrame:")
#     print(sample_data)
#     print("\nCleaned DataFrame:")
#     print(cleaned)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to check for outliers. If None, uses all numeric columns.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    filtered_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)
            filtered_df = filtered_df[mask]
    
    return filtered_df