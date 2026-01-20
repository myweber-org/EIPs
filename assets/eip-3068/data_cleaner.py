
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode.")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has less than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', None, 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData valid: {is_valid}")
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, method='iqr', **kwargs):
    """
    Remove outliers from specified columns.
    Supports 'iqr' and 'zscore' methods.
    """
    data_clean = data.copy()
    
    for col in columns:
        if method == 'iqr':
            outliers = detect_outliers_iqr(data_clean, col, **kwargs)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_clean[col].dropna()))
            outliers = z_scores > kwargs.get('threshold', 3)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        data_clean = data_clean[~outliers]
    
    return data_clean.reset_index(drop=True)

def normalize_column(data, column, method='minmax'):
    """
    Normalize column using specified method.
    Supports 'minmax' and 'standard' normalization.
    """
    if method == 'minmax':
        min_val = data[column].min()
        max_val = data[column].max()
        if max_val - min_val == 0:
            return data[column]
        return (data[column] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        mean_val = data[column].mean()
        std_val = data[column].std()
        if std_val == 0:
            return data[column]
        return (data[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method=None):
    """
    Comprehensive data cleaning pipeline.
    """
    # Remove outliers
    cleaned_data = remove_outliers(data, numeric_columns, method=outlier_method)
    
    # Normalize if requested
    if normalize_method:
        for col in numeric_columns:
            cleaned_data[col] = normalize_column(cleaned_data, col, method=normalize_method)
    
    # Remove any remaining NaN values
    cleaned_data = cleaned_data.dropna().reset_index(drop=True)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate summary statistics for data quality assessment.
    """
    summary = {
        'total_rows': len(data),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    for col in summary['numeric_columns']:
        summary[f'{col}_stats'] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'median': data[col].median()
        }
    
    return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b']
    cleaned = clean_dataset(sample_data, numeric_cols, outlier_method='iqr', normalize_method='standard')
    
    # Get summary
    summary = get_data_summary(cleaned)
    print(f"Original rows: {len(sample_data)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Removed outliers: {len(sample_data) - len(cleaned)}")