
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
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

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'has_data': not df.empty,
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', output_path=None):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    file_path (str): Path to input CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    output_path (str): Optional path to save cleaned data
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif fill_strategy == 'median':
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    elif fill_strategy == 'mode':
        for col in numeric_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
    elif fill_strategy == 'zero':
        df.fillna(0, inplace=True)
    else:
        raise ValueError("Invalid fill_strategy. Choose from: 'mean', 'median', 'mode', 'zero'")
    
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        df[col].fillna('Unknown', inplace=True)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    outliers_removed = len(df) - len(filtered_df)
    print(f"Removed {outliers_removed} outliers from column '{column}'")
    
    return filtered_df

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    columns (list): List of columns to standardize. If None, standardize all numeric columns.
    
    Returns:
    pandas.DataFrame: DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    standardized_df = df.copy()
    
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                standardized_df[col] = (df[col] - mean) / std
            else:
                standardized_df[col] = 0
    
    return standardized_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['X', 'Y', 'Z', np.nan, 'X', 'Y'],
        'D': [0.1, 0.2, 0.3, 0.4, np.nan, 0.6]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='median')
    cleaned_df = remove_outliers_iqr(cleaned_df, 'A')
    standardized_df = standardize_columns(cleaned_df, ['A', 'B', 'D'])
    
    print("\nFinal cleaned and standardized data:")
    print(standardized_df)