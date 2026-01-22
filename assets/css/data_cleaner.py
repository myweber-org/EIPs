import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    Returns filtered DataFrame.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns filtered DataFrame.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to range [0, 1] using Min-Max scaling.
    Returns normalized Series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    Returns normalized Series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    normalized = (data[column] - mean_val) / std_val
    return normalized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    Returns processed DataFrame.
    """
    if strategy == 'drop':
        return data.dropna()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    Applies outlier removal, normalization, and missing value handling.
    Returns cleaned DataFrame.
    """
    cleaned_data = data.copy()
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    for col in numeric_cols:
        if col in cleaned_data.columns:
            if normalize_method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif normalize_method == 'zscore':
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data)
    print("\nCleaned data:")
    print(cleaned)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        method: 'iqr' for interquartile range or 'zscore' for standard deviations
        threshold: Threshold value for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    else:  # zscore
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        filtered_df = df[z_scores < threshold]
    
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return filtered_df