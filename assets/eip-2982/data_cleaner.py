
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nValidation: {is_valid} - {message}")
import numpy as np
import pandas as pd
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
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

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 1000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("Missing values before:", sample_df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b', 'feature_c'])
    cleaner.fill_missing_median()
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Outliers removed: {removed}")
    print("Cleaned data shape:", cleaned_df.shape)
    print("Missing values after:", cleaned_df.isnull().sum().sum())
    print("\nSummary:", summary)
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
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
        return self.df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        self.df = df_clean
        return self.df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self.df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def reset_to_original(self):
        self.df = self.original_df.copy()
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.original_df),
            'current_rows': len(self.df),
            'original_columns': list(self.original_df.columns),
            'current_columns': list(self.df.columns),
            'removed_rows': len(self.original_df) - len(self.df)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    outliers = np.random.randint(0, 100, 5)
    df.loc[outliers, 'feature1'] = np.random.normal(300, 50, 5)
    
    missing = np.random.randint(0, 100, 10)
    df.loc[missing, 'feature2'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_df.shape)
    print("\nRemoving outliers using IQR method...")
    cleaned_df = cleaner.remove_outliers_iqr()
    print("Cleaned data shape:", cleaned_df.shape)
    
    print("\nNormalizing features using min-max scaling...")
    normalized_df = cleaner.normalize_minmax()
    print("Normalized data shape:", normalized_df.shape)
    
    print("\nHandling missing values using mean imputation...")
    filled_df = cleaner.handle_missing_values(strategy='mean')
    print("Data shape after handling missing values:", filled_df.shape)
    
    summary = cleaner.get_summary()
    print("\nData cleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, drop_na=True, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
        drop_na (bool): If True, drop rows with missing values. If False, fill them.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else saves to file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        if drop_na:
            df = df.dropna()
            print(f"Dropped rows with missing values. Removed {missing_before} missing entries")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'mode':
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            elif fill_strategy == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # For non-numeric columns, fill with empty string
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            df[non_numeric_cols] = df[non_numeric_cols].fillna('')
            
            missing_after = df.isnull().sum().sum()
            print(f"Filled {missing_before - missing_after} missing values using {fill_strategy} strategy")
        
        print(f"Final data shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Validation results with status and messages.
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    if df is None or df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty or None')
        return validation_result
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing_cols
            validation_result['messages'].append(f'Missing required columns: {missing_cols}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            validation_result['is_valid'] = False
            validation_result['messages'].append(f'Found {inf_count} infinite values in numeric columns')
    
    return validation_result

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, None, 20.1, 20.1],
        'category': ['A', 'B', None, 'A', 'B', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', drop_na=False, fill_strategy='mean')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print(f"Validation result: {validation}")
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding the threshold percentage.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum allowed missing percentage per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    missing_percent = df.isnull().mean(axis=1)
    return df[missing_percent <= threshold].copy()

def replace_outliers_iqr(df, columns, multiplier=1.5):
    """
    Replace outliers with column boundaries using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers replaced
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    return df_clean

def standardize_columns(df, columns):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_std = df.copy()
    
    for col in columns:
        if col not in df_std.columns:
            continue
            
        mean_val = df_std[col].mean()
        std_val = df_std[col].std()
        
        if std_val > 0:
            df_std[col] = (df_std[col] - mean_val) / std_val
    
    return df_std

def clean_dataset(df, missing_threshold=0.3, outlier_columns=None, standardize_columns_list=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_threshold (float): Threshold for removing rows with missing values
        outlier_columns (list): Columns to process for outliers
        standardize_columns_list (list): Columns to standardize
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if outlier_columns is None:
        outlier_columns = []
    if standardize_columns_list is None:
        standardize_columns_list = []
    
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    
    if outlier_columns:
        df_clean = replace_outliers_iqr(df_clean, outlier_columns)
    
    if standardize_columns_list:
        df_clean = standardize_columns(df_clean, standardize_columns_list)
    
    return df_clean