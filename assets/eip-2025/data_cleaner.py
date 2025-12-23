
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max == col_min:
                self.df[f'{column}_normalized'] = 0.5
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - col_min) / (col_max - col_min)
        
        elif method == 'zscore':
            col_mean = self.df[column].mean()
            col_std = self.df[column].std()
            if col_std == 0:
                self.df[f'{column}_normalized'] = 0
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - col_mean) / col_std
        
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df[f'{column}_normalized']
    
    def fill_missing_values(self, strategy='mean', custom_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'custom' and custom_value is not None:
                    fill_value = custom_value
                else:
                    continue
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df.isnull().sum().sum()
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(1000, 50), 'feature1'] = np.nan
    df.loc[np.random.choice(1000, 20), 'feature2'] = np.nan
    
    cleaner = DataCleaner(df)
    print(f"Initial shape: {cleaner.original_shape}")
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    missing_filled = cleaner.fill_missing_values(strategy='median')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('feature1', method='zscore')
    cleaner.normalize_column('feature2', method='minmax')
    
    summary = cleaner.get_summary()
    print(f"Final shape: {summary['current_rows']} rows, {summary['columns']} columns")
    
    cleaned_df = cleaner.get_cleaned_data()
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print("Data cleaning completed successfully")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
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
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): Required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'column_names': list(df.columns),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return summaryimport pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: pandas DataFrame to clean.
        subset: Column label or sequence of labels to consider for duplicates.
                If None, all columns are used.
        keep: Determines which duplicates to mark.
              'first': Mark duplicates as False except for the first occurrence.
              'last': Mark duplicates as False except for the last occurrence.
              False: Mark all duplicates as True.
    
    Returns:
        Cleaned DataFrame with duplicates removed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows.")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate basic DataFrame structure and required columns.
    
    Args:
        dataframe: pandas DataFrame to validate.
        required_columns: List of column names that must be present.
    
    Returns:
        Boolean indicating if validation passed.
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False
    
    if dataframe.empty:
        print("Warning: DataFrame is empty.")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_column(dataframe, column_name, fill_method='mean'):
    """
    Clean a numeric column by handling missing values.
    
    Args:
        dataframe: pandas DataFrame containing the column.
        column_name: Name of the column to clean.
        fill_method: Method to fill missing values ('mean', 'median', 'zero').
    
    Returns:
        Series with cleaned numeric values.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    column_data = dataframe[column_name].copy()
    
    missing_count = column_data.isna().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values in column '{column_name}'")
        
        if fill_method == 'mean':
            fill_value = column_data.mean()
        elif fill_method == 'median':
            fill_value = column_data.median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
        
        column_data.fillna(fill_value, inplace=True)
        print(f"Filled missing values with {fill_method}: {fill_value}")
    
    return column_dataimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main cleaning function that handles outliers and normalization.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(data, column, check_type='range', min_val=None, max_val=None):
    """
    Validate data based on specified criteria.
    """
    if check_type == 'range':
        if min_val is not None:
            invalid = data[column] < min_val
            if invalid.any():
                print(f"Warning: {invalid.sum()} values below minimum {min_val}")
        if max_val is not None:
            invalid = data[column] > max_val
            if invalid.any():
                print(f"Warning: {invalid.sum()} values above maximum {max_val}")
    return data

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:")
    print(sample_data[['feature1', 'feature2']].describe())
    
    cleaned = clean_dataset(
        sample_data, 
        ['feature1', 'feature2'], 
        outlier_method='iqr', 
        normalize_method='zscore'
    )
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned statistics:")
    print(cleaned[['feature1_standardized', 'feature2_standardized']].describe())
    
    return cleaned

if __name__ == "__main__":
    result = example_usage()