
import re
import unicodedata

def clean_text(text, remove_digits=False, keep_case=False):
    """
    Clean and normalize input text by removing extra whitespace,
    optionally removing digits, and optionally preserving case.
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_digits:
        text = re.sub(r'\d+', '', text)

    if not keep_case:
        text = text.lower()

    return text

def tokenize_text(text, pattern=r'\w+'):
    """
    Tokenize text using a regex pattern.
    Default pattern matches alphanumeric words.
    """
    tokens = re.findall(pattern, text)
    return tokens

def remove_stopwords(tokens, stopwords=None):
    """
    Remove stopwords from a list of tokens.
    If no stopwords provided, uses a minimal default set.
    """
    if stopwords is None:
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at'}

    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    return filtered_tokens
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
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    }
    
    df = pd.DataFrame(data)
    df.iloc[10:15, 0] = np.nan
    df.iloc[20:25, 1] = np.nan
    
    cleaner = DataCleaner(df)
    print(f"Initial shape: {cleaner.get_summary()['original_rows']} rows")
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing_median()
    cleaner.standardize_zscore(['feature_a', 'feature_b'])
    cleaner.normalize_minmax(['feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Final shape: {summary['cleaned_rows']} rows")
    print(f"Missing values: {summary['missing_values']}")
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())