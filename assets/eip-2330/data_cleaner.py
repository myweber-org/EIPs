import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Error cleaning column {column}: {e}")
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_rows = initial_rows - df_clean.shape[0]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Log cleaning results
    print(f"Removed {removed_rows} duplicate rows.")
    print(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    """
    df_filled = df.copy()
    
    # Select numeric columns
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    elif strategy == 'median':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].median())
    elif strategy == 'zero':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)
    
    missing_count = df_filled[numeric_cols].isnull().sum().sum()
    print(f"Missing values after filling: {missing_count}")
    
    return df_filled

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Age': [25, 30, 25, 35, np.nan],
        'Score': [85, 90, 85, 95, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\n")
    
    filled_df = handle_missing_values(cleaned_df, strategy='mean')
    print("\nDataset after handling missing values:")
    print(filled_df)import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows,
                       'fill' to fill with column mean (numeric) or mode (categorical).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill':
        for column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
            else:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown')
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
            print("Filled missing values with mode")
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 30, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'], min_rows=1)
    print(f"\nValidation: {message}")
import numpy as np
import pandas as pd
from scipy import stats

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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
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
    Main function to clean dataset by removing outliers and normalizing numeric columns
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
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common issues
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_report['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum().sum()
        validation_report['total_missing_values'] = missing_values
    
    if check_duplicates:
        duplicate_rows = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_rows
    
    return validation_reportimport numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
                
        return clean_df.reset_index(drop=True)
    
    def impute_missing_values(self, strategy='median', fill_value=None):
        imputed_df = self.df.copy()
        
        for col in self.numeric_columns:
            if imputed_df[col].isnull().any():
                if strategy == 'mean':
                    imputed_df[col].fillna(imputed_df[col].mean(), inplace=True)
                elif strategy == 'median':
                    imputed_df[col].fillna(imputed_df[col].median(), inplace=True)
                elif strategy == 'mode':
                    imputed_df[col].fillna(imputed_df[col].mode()[0], inplace=True)
                elif strategy == 'constant' and fill_value is not None:
                    imputed_df[col].fillna(fill_value, inplace=True)
                    
        return imputed_df
    
    def detect_anomalies_zscore(self, threshold=3):
        anomalies = {}
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            anomaly_indices = np.where(z_scores > threshold)[0]
            if len(anomaly_indices) > 0:
                anomalies[col] = {
                    'indices': anomaly_indices.tolist(),
                    'values': self.df[col].iloc[anomaly_indices].tolist(),
                    'count': len(anomaly_indices)
                }
        return anomalies
    
    def get_summary(self):
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': list(self.numeric_columns),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(60, 15, 100),
        'pressure': np.random.normal(1013, 10, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'temperature'] = np.nan
    df.loc[95, 'humidity'] = 150
    df.loc[96, 'pressure'] = 2000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nDetected Anomalies:")
    anomalies = cleaner.detect_anomalies_zscore()
    for col, details in anomalies.items():
        print(f"{col}: {details['count']} anomalies")
    
    cleaned_df = cleaner.remove_outliers_iqr()
    imputed_df = cleaner.impute_missing_values(strategy='median')
    
    print(f"\nOriginal shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Imputed shape: {imputed_df.shape}")
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fillna_method (str or None): Method to fill missing values:
            None: Drop rows with any missing values
            'mean': Fill with column mean (numeric only)
            'median': Fill with column median (numeric only)
            'mode': Fill with most frequent value
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fillna_method is None:
        cleaned_df = cleaned_df.dropna()
    elif fillna_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fillna_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fillna_method == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        dict: Validation results with keys:
            'is_valid': Boolean indicating if validation passed
            'message': Description of validation result
            'missing_columns': List of missing required columns
            'row_count': Number of rows
            'column_count': Number of columns
    """
    result = {
        'is_valid': True,
        'message': 'Validation passed',
        'missing_columns': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check for required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            result['is_valid'] = False
            result['missing_columns'] = missing
            result['message'] = f'Missing required columns: {missing}'
    
    # Check for empty DataFrame
    if df.empty:
        result['is_valid'] = False
        result['message'] = 'DataFrame is empty'
    
    return result

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'numeric_stats': {}
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, None, 28, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, fillna_method='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the data
    validation = validate_dataframe(cleaned, required_columns=['id', 'name', 'age', 'score'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    print("\n" + "="*50 + "\n")
    
    # Get summary statistics
    summary = get_data_summary(cleaned)
    print("Data Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Missing Values: {summary['missing_values']}")
    print("\nNumeric Statistics:")
    for col, stats in summary['numeric_stats'].items():
        print(f"{col}: {stats}")import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda x: isinstance(x, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda x: not x.empty, "DataFrame cannot be empty"),
        (lambda x: x.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check_func, error_msg in required_checks:
        if not check_func(df):
            raise ValueError(error_msg)
    return True
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (str or dict): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
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
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values remain in the dataset")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'age': [25, 30, 25, None, 35],
        'score': [85, 90, 85, 88, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, fill_missing={'age': df['age'].mean(), 'name': 'Unknown'})
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['name', 'age'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop'):
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    handle_nulls (str): How to handle null values - 'drop', 'fill_mean', 'fill_median', or 'fill_zero'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} rows with null values")
    elif handle_nulls == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        print("Filled numeric nulls with column means")
    elif handle_nulls == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print("Filled numeric nulls with column medians")
    elif handle_nulls == 'fill_zero':
        cleaned_df = cleaned_df.fillna(0)
        print("Filled all nulls with zeros")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    validation_results['summary']['total_rows'] = len(df)
    validation_results['summary']['total_columns'] = len(df.columns)
    validation_results['summary']['null_count'] = df.isnull().sum().sum()
    validation_results['summary']['duplicate_rows'] = df.duplicated().sum()
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    if validation_results['summary']['null_count'] > 0:
        validation_results['warnings'].append(f'Dataset contains {validation_results["summary"]["null_count"]} null values')
    
    if validation_results['summary']['duplicate_rows'] > 0:
        validation_results['warnings'].append(f'Dataset contains {validation_results["summary"]["duplicate_rows"]} duplicate rows')
    
    return validation_results

def sample_data_for_testing():
    """
    Create a sample DataFrame for testing the cleaning functions.
    
    Returns:
    pd.DataFrame: Sample DataFrame with various data quality issues
    """
    data = {
        'id': [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David', 'Eve', 'Frank', None, 'Helen', 'Ivan'],
        'age': [25, 30, 35, 35, 40, None, 50, 55, 60, 65],
        'score': [85.5, 90.0, 78.5, 78.5, 92.0, 88.5, None, 76.0, 81.5, 94.0],
        'department': ['Sales', 'Marketing', 'IT', 'IT', 'HR', 'Finance', 'Sales', 'Marketing', None, 'IT']
    }
    
    return pd.DataFrame(data)
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: 'first', 'last', or False to drop all duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics before and after cleaning
    """
    original_stats = {}
    cleaned_stats = {}
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_stats[column] = calculate_statistics(df, column)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            cleaned_stats[column] = calculate_statistics(cleaned_df, column)
    
    stats_comparison = {
        'original': original_stats,
        'cleaned': cleaned_stats,
        'rows_removed': len(df) - len(cleaned_df)
    }
    
    return cleaned_df, stats_comparison

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[np.random.choice(sample_df.index, 50), 'A'] = 500
    
    cleaned_df, stats = clean_dataset(sample_df, ['A', 'B', 'C'])
    
    print(f"Original rows: {len(sample_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Rows removed: {stats['rows_removed']}")
    print(f"Column A mean - Original: {stats['original']['A']['mean']:.2f}, Cleaned: {stats['cleaned']['A']['mean']:.2f}")