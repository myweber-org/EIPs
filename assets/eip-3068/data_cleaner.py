
import pandas as pd

def clean_dataframe(df, fill_strategy='drop', column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    fill_strategy (str): Strategy for handling null values. 
                         Options: 'drop' to remove rows, 'fill' to fill with column mean (numeric) or mode (categorical).
    column_case (str): Target case for column names. Options: 'lower', 'upper', 'title'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Standardize column names
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    # Handle missing values
    if fill_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif fill_strategy == 'fill':
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else '', inplace=True)
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if df.empty:
        validation_result['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        validation_result['warnings'].append('Duplicate column names detected')
    
    return validation_result

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 30, 35],
        'Score': [85.5, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, fill_strategy='fill', column_case='lower')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
    print("Validation Result:")
    print(validation)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def zscore_normalize(dataframe, columns=None):
    """
    Normalize specified columns using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            
            if std_val > 0:
                result_df[f'{col}_normalized'] = (result_df[col] - mean_val) / std_val
            else:
                result_df[f'{col}_normalized'] = 0
    
    return result_df

def minmax_normalize(dataframe, columns=None, feature_range=(0, 1)):
    """
    Normalize specified columns using min-max normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    min_range, max_range = feature_range
    
    for col in columns:
        if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
            min_val = result_df[col].min()
            max_val = result_df[col].max()
            
            if max_val > min_val:
                normalized = (result_df[col] - min_val) / (max_val - min_val)
                result_df[f'{col}_scaled'] = normalized * (max_range - min_range) + min_range
            else:
                result_df[f'{col}_scaled'] = min_range
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'constant', 'drop')
        columns: list of columns to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if result_df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(result_df[col]):
                fill_value = result_df[col].mean()
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(result_df[col]):
                fill_value = result_df[col].median()
            elif strategy == 'mode':
                fill_value = result_df[col].mode()[0] if not result_df[col].mode().empty else None
            elif strategy == 'constant':
                fill_value = 0 if pd.api.types.is_numeric_dtype(result_df[col]) else 'missing'
            elif strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
                continue
            else:
                continue
                
            result_df[col] = result_df[col].fillna(fill_value)
    
    return result_df

def validate_dataframe(dataframe, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_columns: list of columns that should be numeric
    
    Returns:
        tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns 
                      if col in dataframe.columns 
                      and not pd.api.types.is_numeric_dtype(dataframe[col])]
        if non_numeric:
            return False, f"Non-numeric columns found: {non_numeric}"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame validation passed"

def get_data_summary(dataframe):
    """
    Generate comprehensive summary statistics for DataFrame.
    
    Args:
        dataframe: pandas DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'shape': dataframe.shape,
        'columns': dataframe.columns.tolist(),
        'dtypes': dataframe.dtypes.to_dict(),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            '25%': dataframe[col].quantile(0.25),
            '50%': dataframe[col].median(),
            '75%': dataframe[col].quantile(0.75),
            'max': dataframe[col].max(),
            'skewness': dataframe[col].skew(),
            'kurtosis': dataframe[col].kurtosis()
        }
    
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = dataframe[col].value_counts()
        summary['categorical_stats'][col] = {
            'unique_count': dataframe[col].nunique(),
            'top_value': value_counts.index[0] if not value_counts.empty else None,
            'top_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'value_counts': value_counts.head(10).to_dict()
        }
    
    return summary