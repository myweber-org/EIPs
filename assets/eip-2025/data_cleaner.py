
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        return self.df.head()
    
    def check_missing_values(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_summary = self.df.isnull().sum()
        missing_percentage = (missing_summary / len(self.df)) * 100
        
        missing_report = pd.DataFrame({
            'missing_count': missing_summary,
            'missing_percentage': missing_percentage
        })
        
        return missing_report[missing_report['missing_count'] > 0]
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df_cleaned = self.df.copy()
        
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_cleaned[column]):
                    df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                elif strategy == 'mode':
                    df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
                elif strategy == 'ffill':
                    df_cleaned[column].fillna(method='ffill', inplace=True)
                elif strategy == 'bfill':
                    df_cleaned[column].fillna(method='bfill', inplace=True)
                elif fill_value is not None:
                    df_cleaned[column].fillna(fill_value, inplace=True)
                else:
                    df_cleaned[column].fillna(0, inplace=True)
        
        return df_cleaned
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(self.df)
        
        return removed_count
    
    def save_cleaned_data(self, output_path, format='csv'):
        if self.df is None:
            raise ValueError("No data to save. Load and clean data first.")
        
        output_path = Path(output_path)
        
        if format == 'csv':
            self.df.to_csv(output_path, index=False)
        elif format == 'excel':
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output format. Use 'csv' or 'excel'.")
        
        return f"Data saved to {output_path}"
    
    def get_summary_statistics(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        summary = self.df[numeric_cols].describe()
        
        return summary

def process_csv_file(input_file, output_file):
    cleaner = DataCleaner(input_file)
    cleaner.load_data()
    
    missing_report = cleaner.check_missing_values()
    print("Missing values report:")
    print(missing_report)
    
    cleaned_data = cleaner.handle_missing_values(strategy='mean')
    cleaner.df = cleaned_data
    
    duplicates_removed = cleaner.remove_duplicates()
    print(f"Removed {duplicates_removed} duplicate rows")
    
    cleaner.save_cleaned_data(output_file)
    
    summary = cleaner.get_summary_statistics()
    print("\nSummary statistics:")
    print(summary)
    
    return cleaner.df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10.5, 20.3, np.nan, 15.7, 25.1, np.nan, 18.9, 22.4, 19.6, 21.8],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'score': [85, 92, 78, np.nan, 88, 95, 91, 87, 84, 90]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    result = process_csv_file('test_data.csv', 'cleaned_data.csv')
    print("\nCleaned data preview:")
    print(result.head())
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to check for missing values.
                                  If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed.
    """
    if columns is None:
        return df.dropna()
    else:
        return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with their mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to fill missing values
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame containing only the outlier rows.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def standardize_column(df, column):
    """
    Standardize a column using z-score normalization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_standardized = df.copy()
    mean_val = df_standardized[column].mean()
    std_val = df_standardized[column].std()
    
    if std_val > 0:
        df_standardized[column] = (df_standardized[column] - mean_val) / std_val
    
    return df_standardized

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including missing values and basic statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing data summary.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
    }
    return summaryimport pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    df_cleaned = df.copy()
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
        outliers = z_scores > 3
        df_cleaned.loc[outliers, col] = np.nan
        
    df_cleaned = df_cleaned.dropna()
    print(f"After outlier removal: {df_cleaned.shape}")
    
    for col in numeric_cols:
        if col in df_cleaned.columns:
            col_min = df_cleaned[col].min()
            col_max = df_cleaned[col].max()
            if col_max > col_min:
                df_cleaned[col] = (df_cleaned[col] - col_min) / (col_max - col_min)
    
    return df_cleaned

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("sample_data.csv")
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    print("Data cleaning complete. Saved to cleaned_data.csv")