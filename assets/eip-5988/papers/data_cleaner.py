
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    processed_df = df.copy()
    for col in processed_df.columns:
        if processed_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0]
            else:
                fill_value = 0
            processed_df[col].fillna(fill_value, inplace=True)
    return processed_df

def validate_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True

def process_dataset(file_path, outlier_cols=None, normalize_cols=None):
    try:
        df = pd.read_csv(file_path)
        validate_dataframe(df)
        
        df = handle_missing_values(df)
        
        if outlier_cols:
            df = remove_outliers_iqr(df, outlier_cols)
        
        if normalize_cols:
            df = normalize_minmax(df, normalize_cols)
        
        return df
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        if normalize_method == 'minmax':
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f"{column}_normalized"] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common issues.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        check_missing: flag to check for missing values
        check_duplicates: flag to check for duplicate rows
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_results['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum().to_dict()
        validation_results['missing_values'] = {k: v for k, v in missing_values.items() if v > 0}
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    return validation_results

def example_usage():
    """
    Example usage of the data cleaning utilities.
    """
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Original data shape:", sample_data.shape)
    
    validation = validate_data(sample_data, required_columns=['id', 'feature_a', 'feature_b'])
    print("Validation results:", validation)
    
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b'],
        outlier_method='iqr',
        normalize_method='zscore'
    )
    
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned data columns:", cleaned.columns.tolist())
    
    return cleaned

if __name__ == "__main__":
    result = example_usage()
    print("\nData cleaning completed successfully.")