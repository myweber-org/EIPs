
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns, method='iqr', threshold=1.5):
        outlier_indices = set()
        for col in columns:
            if method == 'iqr':
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.update(indices)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                indices = self.df[col].dropna().index[z_scores > threshold].tolist()
                outlier_indices.update(indices)
        
        self.df = self.df.drop(index=list(outlier_indices)).reset_index(drop=True)
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[f'{column}_standardized'] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
        
        self.df[column] = self.df[column].fillna(fill_value)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_added = cleaned_shape[1] - self.original_shape[1]
        
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_added': cols_added
        }
        return summary

def process_dataset(file_path, numerical_cols, categorical_cols=None):
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    if numerical_cols:
        cleaner.remove_outliers(numerical_cols, method='iqr', threshold=1.5)
        for col in numerical_cols:
            cleaner.normalize_column(col, method='minmax')
            cleaner.fill_missing(col, method='mean')
    
    if categorical_cols:
        for col in categorical_cols:
            cleaner.fill_missing(col, method='mode')
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()
def deduplicate_list(original_list):
    seen = set()
    deduplicated = []
    for item in original_list:
        if item not in seen:
            seen.add(item)
            deduplicated.append(item)
    return deduplicated
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode' and self.categorical_columns:
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        elif strategy == 'drop':
            self.df = self.df.dropna()
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self

    def get_cleaned_data(self):
        return self.df

def load_and_clean_data(filepath, missing_strategy='mean', remove_outliers=True):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    if remove_outliers:
        cleaner.remove_outliers_iqr()
    return cleaner.get_cleaned_data()
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    """
    clean_data = data.copy()
    for col in columns:
        outliers = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    Returns normalized DataFrame.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    Returns standardized DataFrame.
    """
    standardized_data = data.copy()
    for col in columns:
        mean_val = standardized_data[col].mean()
        std_val = standardized_data[col].std()
        standardized_data[col] = (standardized_data[col] - mean_val) / std_val
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    processed_data = data.copy()
    
    if strategy == 'drop':
        return processed_data.dropna(subset=columns)
    
    for col in columns:
        if processed_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[col].mean()
            elif strategy == 'median':
                fill_value = processed_data[col].median()
            elif strategy == 'mode':
                fill_value = processed_data[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            processed_data[col] = processed_data[col].fillna(fill_value)
    
    return processed_data

def clean_dataset(data, numeric_columns, outlier_threshold=1.5, 
                  normalization='minmax', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy, 
                                         columns=numeric_columns)
    
    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, numeric_columns, 
                                   threshold=outlier_threshold)
    
    # Apply normalization
    if normalization == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, numeric_columns)
    elif normalization == 'zscore':
        cleaned_data = standardize_zscore(cleaned_data, numeric_columns)
    
    return cleaned_data