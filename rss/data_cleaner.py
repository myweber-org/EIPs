
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def remove_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.data[column]))
        return self.data[z_scores < threshold]
    
    def normalize_minmax(self, column):
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        self.data[column + '_normalized'] = (self.data[column] - min_val) / (max_val - min_val)
        return self.data
    
    def standardize_zscore(self, column):
        mean_val = self.data[column].mean()
        std_val = self.data[column].std()
        self.data[column + '_standardized'] = (self.data[column] - mean_val) / std_val
        return self.data
    
    def handle_missing_mean(self, column):
        self.data[column].fillna(self.data[column].mean(), inplace=True)
        return self.data
    
    def handle_missing_median(self, column):
        self.data[column].fillna(self.data[column].median(), inplace=True)
        return self.data
    
    def get_summary(self):
        summary = {
            'original_shape': self.data.shape,
            'cleaned_shape': self.cleaned_data.shape if self.cleaned_data is not None else None,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        return summary

def clean_dataset(df, outlier_method='iqr', normalize=True):
    cleaner = DataCleaner(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            df = cleaner.remove_outliers_iqr(col)
        elif outlier_method == 'zscore':
            df = cleaner.remove_outliers_zscore(col)
        
        if normalize:
            df = cleaner.normalize_minmax(col)
    
    cleaner.cleaned_data = df
    return cleanerimport pandas as pd

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
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
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
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"Validation: {is_valid} - {message}")
import pandas as pd
import numpy as np

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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    # Add some outliers
    sample_data['A'][:50] = np.random.uniform(300, 500, 50)
    sample_data['B'][:30] = np.random.uniform(400, 600, 30)
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    
    # Calculate statistics
    print("\nOriginal statistics:")
    print(df.describe())
    
    print("\nCleaned statistics:")
    print(cleaned_df.describe())