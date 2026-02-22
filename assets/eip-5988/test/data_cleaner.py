
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if self.df[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                self.df[col].fillna(fill_value, inplace=True)
            else:
                self.df[col].fillna('Unknown', inplace=True)
        return self

    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            if method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self

    def get_cleaned_data(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        return self.df

    def save_cleaned_data(self, filepath):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, factor=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    factor (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if not np.issubdtype(result_df[col].dtype, np.number):
            continue
            
        col_min = result_df[col].min()
        col_max = result_df[col].max()
        
        if col_max == col_min:
            result_df[col] = 0.5
        else:
            result_df[col] = (result_df[col] - col_min) / (col_max - col_min)
    
    return result_df

def z_score_normalize(dataframe, columns=None):
    """
    Normalize specified columns using z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with z-score normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if not np.issubdtype(result_df[col].dtype, np.number):
            continue
            
        mean_val = result_df[col].mean()
        std_val = result_df[col].std()
        
        if std_val == 0:
            result_df[col] = 0
        else:
            result_df[col] = (result_df[col] - mean_val) / std_val
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'constant')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if result_df[col].isnull().sum() == 0:
            continue
        
        if strategy == 'mean' and np.issubdtype(result_df[col].dtype, np.number):
            fill_value = result_df[col].mean()
        elif strategy == 'median' and np.issubdtype(result_df[col].dtype, np.number):
            fill_value = result_df[col].median()
        elif strategy == 'mode':
            fill_value = result_df[col].mode()[0] if not result_df[col].mode().empty else None
        elif strategy == 'constant':
            fill_value = 0 if np.issubdtype(result_df[col].dtype, np.number) else ''
        else:
            continue
        
        if fill_value is not None:
            result_df[col] = result_df[col].fillna(fill_value)
    
    return result_df

def clean_dataset(dataframe, outlier_columns=None, normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns to remove outliers from. If None, skip outlier removal.
    normalize_method (str): Normalization method ('minmax', 'zscore', or None)
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    if missing_strategy:
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df)
    elif normalize_method == 'zscore':
        cleaned_df = z_score_normalize(cleaned_df)
    
    return cleaned_df

def get_data_summary(dataframe):
    """
    Generate a summary of the DataFrame including statistics and missing values.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing data summary
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
            'max': dataframe[col].max()
        }
    
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': dataframe[col].nunique(),
            'top_value': dataframe[col].mode()[0] if not dataframe[col].mode().empty else None,
            'top_count': dataframe[col].value_counts().iloc[0] if not dataframe[col].value_counts().empty else 0
        }
    
    return summary
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
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

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11, 14, 13, 12, 14, 200]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned data:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary statistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")