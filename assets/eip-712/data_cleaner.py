
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling
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
    Standardize column using z-score normalization
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
    
    data_copy = data.copy()
    
    for column in columns:
        if column not in data_copy.columns:
            continue
            
        if data_copy[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[column].mean()
            elif strategy == 'median':
                fill_value = data_copy[column].median()
            elif strategy == 'mode':
                fill_value = data_copy[column].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[column] = data_copy[column].fillna(fill_value)
    
    return data_copy

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.uniform(0, 1, 100),
        'feature_c': np.random.exponential(2, 100)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(100, size=10, replace=False)
    for idx in indices:
        df.loc[idx, 'feature_a'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("Missing values:", sample_data.isnull().sum())
    
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    print("After handling missing values:", cleaned_data.shape)
    
    normalized_feature = normalize_minmax(cleaned_data, 'feature_b')
    print("Normalized feature_b range:", normalized_feature.min(), normalized_feature.max())
    
    standardized_feature = standardize_zscore(cleaned_data, 'feature_a')
    print("Standardized feature_a mean:", standardized_feature.mean())
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, missing_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        missing_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype in ['int64', 'float64']:
                    if missing_strategy == 'mean':
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                    elif missing_strategy == 'median':
                        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                    elif missing_strategy == 'zero':
                        cleaned_df[column] = cleaned_df[column].fillna(0)
                elif cleaned_df[column].dtype == 'object':
                    if missing_strategy == 'mode':
                        mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else ''
                        cleaned_df[column] = cleaned_df[column].fillna(mode_value)
                    else:
                        cleaned_df[column] = cleaned_df[column].fillna('')
            
            print(f"Filled missing values using '{missing_strategy}' strategy")
    
    return cleaned_df

def validate_dataset(df):
    """
    Validate dataset for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'column_stats': {}
    }
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            validation_results['column_stats'][column] = {
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean(),
                'std': df[column].std()
            }
        elif df[column].dtype == 'object':
            validation_results['column_stats'][column] = {
                'unique_values': df[column].nunique(),
                'most_common': df[column].mode()[0] if not df[column].mode().empty else None
            }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, missing_strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df)
    print("\nValidation results:")
    for key, value in validation.items():
        if key != 'column_stats':
            print(f"{key}: {value}")
    
    print("\nColumn statistics:")
    for column, stats in validation['column_stats'].items():
        print(f"\n{column}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value}")