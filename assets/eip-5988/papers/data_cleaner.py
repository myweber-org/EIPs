
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, process all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'original_rows': len(df),
        'cleaned_rows': None,
        'removed_rows': None,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns)
    }
    return summary

def process_dataset(file_path, output_path=None):
    """
    Load, clean, and optionally save a dataset.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str): Optional path to save cleaned data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read file '{file_path}': {e}")
    
    original_summary = get_data_summary(df)
    print(f"Original dataset: {original_summary['original_rows']} rows")
    
    cleaned_df = clean_numeric_data(df)
    
    cleaned_summary = get_data_summary(cleaned_df)
    cleaned_summary['original_rows'] = original_summary['original_rows']
    cleaned_summary['removed_rows'] = original_summary['original_rows'] - cleaned_summary['original_rows']
    
    print(f"Cleaned dataset: {cleaned_summary['original_rows']} rows")
    print(f"Removed {cleaned_summary['removed_rows']} outlier rows")
    
    if output_path:
        try:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save to '{output_path}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data with outliers:")
    print(df)
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned data:")
    print(cleaned_df)
def deduplicate_list(sequence):
    seen = set()
    deduplicated = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            deduplicated.append(item)
    return deduplicated
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
        
    def normalize_data(self, method='minmax', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_norm = self.df.copy()
        
        if method == 'minmax':
            for col in columns:
                if col in df_norm.columns:
                    min_val = df_norm[col].min()
                    max_val = df_norm[col].max()
                    if max_val > min_val:
                        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                        
        elif method == 'zscore':
            for col in columns:
                if col in df_norm.columns:
                    mean_val = df_norm[col].mean()
                    std_val = df_norm[col].std()
                    if std_val > 0:
                        df_norm[col] = (df_norm[col] - mean_val) / std_val
        
        self.df = df_norm
        return self
        
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                else:
                    fill_value = 0
                    
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]