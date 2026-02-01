import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def process_dataframe(df, required_cols=None, numeric_cols=None):
    """
    Main function to process and clean DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        required_cols (list, optional): Required columns for validation.
        numeric_cols (list, optional): Numeric columns to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame or None if validation fails.
    """
    if required_cols and not validate_dataframe(df, required_cols):
        return None
    
    df_cleaned = remove_duplicates(df)
    
    if numeric_cols:
        df_cleaned = clean_numeric_columns(df_cleaned, numeric_cols)
    
    return df_cleanedimport pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode' and self.categorical_columns.any():
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns.any():
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[mask]
        elif method == 'iqr' and self.numeric_columns.any():
            Q1 = self.df[self.numeric_columns].quantile(0.25)
            Q3 = self.df[self.numeric_columns].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((self.df[self.numeric_columns] < (Q1 - 1.5 * IQR)) | 
                     (self.df[self.numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
            self.df = self.df[mask]
        return self

    def normalize_data(self, method='minmax'):
        if method == 'minmax' and self.numeric_columns.any():
            self.df[self.numeric_columns] = (
                self.df[self.numeric_columns] - self.df[self.numeric_columns].min()
            ) / (self.df[self.numeric_columns].max() - self.df[self.numeric_columns].min())
        elif method == 'standard' and self.numeric_columns.any():
            self.df[self.numeric_columns] = (
                self.df[self.numeric_columns] - self.df[self.numeric_columns].mean()
            ) / self.df[self.numeric_columns].std()
        return self

    def get_cleaned_data(self):
        return self.df

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    cleaned_df = (
        cleaner
        .handle_missing_values(strategy='mean')
        .remove_outliers(method='zscore', threshold=3)
        .normalize_data(method='minmax')
        .get_cleaned_data()
    )
    return cleaned_df