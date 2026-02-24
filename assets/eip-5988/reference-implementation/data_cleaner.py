
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
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

def sample_data_cleaning():
    """
    Example usage of the data cleaning functions.
    """
    data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, 20, 20, None, 40],
        'category': ['A', 'B', 'B', 'C', None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    is_valid, message = validate_dataframe(cleaned, ['id', 'value'])
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")

if __name__ == "__main__":
    sample_data_cleaning()
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                mask = z_scores < threshold
                clean_df = clean_df[mask]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        return normalized_df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = filled_df[col].mean()
                filled_df[col] = filled_df[col].fillna(mean_val)
        
        return filled_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                median_val = filled_df[col].median()
                filled_df[col] = filled_df[col].fillna(median_val)
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    data['feature_a'][np.random.choice(100, 5)] = np.nan
    data['feature_b'][np.random.choice(100, 3)] = np.nan
    
    outliers = np.random.choice(100, 5)
    data['feature_a'][outliers] = data['feature_a'][outliers] * 5
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data summary:")
    print(cleaner.get_summary())
    
    cleaned_df = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"\nAfter IQR outlier removal: {cleaned_df.shape}")
    
    normalized_df = cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print(f"\nAfter min-max normalization: {normalized_df.shape}")
    
    filled_df = cleaner.fill_missing_mean(['feature_a', 'feature_b'])
    print(f"\nAfter filling missing values: {filled_df.shape}")import numpy as np
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
        return df[column].apply(lambda x: 0.0)
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
        lambda x: isinstance(x, pd.DataFrame),
        lambda x: not x.empty,
        lambda x: x.isnull().sum().sum() == 0
    ]
    for check in required_checks:
        if not check(df):
            return False
    return True

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2.0, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50)
    }
    df = pd.DataFrame(sample_data)
    print("Original shape:", df.shape)
    
    if validate_dataframe(df):
        cleaned = clean_dataset(df, ['feature_a', 'feature_b'])
        print("Cleaned shape:", cleaned.shape)
        print("Cleaned stats:")
        print(cleaned[['feature_a', 'feature_b']].describe())
    else:
        print("Data validation failed")