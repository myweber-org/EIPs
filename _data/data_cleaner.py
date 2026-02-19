
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def calculate_statistics(df):
    stats_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': stats.skew(df[col].dropna()),
            'kurtosis': stats.kurtosis(df[col].dropna())
        }
    return pd.DataFrame(stats_dict).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1, 200)
    })
    
    print("Original dataset shape:", sample_data.shape)
    print("Original statistics:")
    print(calculate_statistics(sample_data))
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned_data = clean_dataset(sample_data, numeric_cols)
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("Cleaned statistics:")
    print(calculate_statistics(cleaned_data[numeric_cols]))
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing values. If None, uses column mean for numeric, mode for categorical.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.columns:
            missing_count = cleaned_df[column].isnull().sum()
            if missing_count > 0:
                if cleaned_df[column].dtype in ['int64', 'float64']:
                    fill_val = fill_value if fill_value is not None else cleaned_df[column].mean()
                    cleaned_df[column] = cleaned_df[column].fillna(fill_val)
                    print(f"Filled {missing_count} missing values in '{column}' with {fill_val}")
                else:
                    fill_val = fill_value if fill_value is not None else cleaned_df[column].mode()[0]
                    cleaned_df[column] = cleaned_df[column].fillna(fill_val)
                    print(f"Filled {missing_count} missing values in '{column}' with '{fill_val}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum()
        }
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("DataFrame is empty")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        validation_results['issues'].append(f"Found {duplicate_rows} duplicate rows")
    
    return validation_results

def sample_data_cleaning():
    """Example usage of the data cleaning functions."""
    np.random.seed(42)
    
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 40, 40, 45],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(df, required_columns=['id', 'name', 'age'])
    print("Validation Results:")
    print(validation)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = sample_data_cleaning()