import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
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
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            # Remove outliers
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            
            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            cleaned_df = cleaned_df[mask]
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate data structure and content
    """
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            raise ValueError(f"DataFrame contains NaN values in columns: {nan_columns}")
    
    return True

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    # Add some outliers
    data['feature_a'][:50] = np.random.normal(300, 10, 50)
    data['feature_b'][:30] = np.random.normal(500, 20, 30)
    
    return pd.DataFrame(data)import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
        outliers = z_scores > 3
        df_clean.loc[outliers, col] = np.nan
        print(f"Removed {outliers.sum()} outliers from {col}")
    
    df_clean = df_clean.dropna(subset=numeric_cols)
    print(f"Shape after outlier removal: {df_clean.shape}")
    
    for col in numeric_cols:
        col_min = df_clean[col].min()
        col_max = df_clean[col].max()
        if col_max > col_min:
            df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)
    
    return df_clean

if __name__ == "__main__":
    cleaned = load_and_clean_data("sample_data.csv")
    cleaned.to_csv("cleaned_data.csv", index=False)
    print("Data cleaning complete. Saved to cleaned_data.csv")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Strategy to fill missing values. 
            Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            raise ValueError("Invalid fill_missing value. Use 'mean', 'median', 'mode', or a dictionary.")
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values remain in the dataset")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'x', 'z', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    print("\nValidation result:", validate_dataframe(cleaned, ['A', 'B']))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: column name to process
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(dataframe[column].dropna()))
    
    filtered_df = dataframe[(z_scores < threshold) | 
                           (dataframe[column].isna())]
    
    return filtered_df

def normalize_minmax(dataframe, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(dataframe, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    
    standardized = (dataframe[column] - mean_val) / std_val
    return standardized

def clean_dataset(dataframe, numeric_columns=None, outlier_method='iqr', 
                  normalize_method='minmax', outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        numeric_columns: list of numeric columns to process (None for all numeric)
        outlier_method: 'iqr' or 'zscore' for outlier removal
        normalize_method: 'minmax' or 'zscore' for normalization
        outlier_threshold: threshold parameter for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column, outlier_threshold)
        
        if normalize_method == 'minmax':
            cleaned_df[f'{column}_normalized'] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[f'{column}_standardized'] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, 
                       numeric_check=True, null_threshold=0.5):
    """
    Validate DataFrame structure and data quality.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_check: whether to check for numeric columns
        null_threshold: maximum allowed proportion of null values
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns 
                          if col not in dataframe.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Missing required columns: {missing_columns}"
            )
    
    if numeric_check:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            validation_results['warnings'].append(
                "No numeric columns found in DataFrame"
            )
    
    for column in dataframe.columns:
        null_proportion = dataframe[column].isnull().mean()
        if null_proportion > null_threshold:
            validation_results['warnings'].append(
                f"Column '{column}' has {null_proportion:.1%} null values"
            )
    
    return validation_results