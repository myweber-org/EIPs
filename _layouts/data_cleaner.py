
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
                
        self.df = df_clean.reset_index(drop=True)
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        return self
        
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    else:
                        df_normalized[col] = 0
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    if iqr != 0:
                        df_normalized[col] = (self.df[col] - median) / iqr
                    else:
                        df_normalized[col] = 0
                        
        self.df = df_normalized
        print(f"Applied {method} normalization to {len(columns)} columns")
        return self
        
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                else:
                    fill_value = strategy
                    
                df_filled[col] = self.df[col].fillna(fill_value)
                
        self.df = df_filled
        print(f"Handled missing values using {strategy} strategy")
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
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(100, 5), 'feature_a'] = np.nan
    df.loc[np.random.choice(100, 3), 'feature_b'] = np.nan
    
    outliers = np.random.choice(100, 10)
    df.loc[outliers, 'feature_a'] = df['feature_a'].mean() + 5 * df['feature_a'].std()
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .handle_missing_values(strategy='median')
                 .remove_outliers_iqr(threshold=1.5)
                 .normalize_data(method='zscore')
                 .get_cleaned_data())
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data preview:")
    print(cleaned_df.head())
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep: 'first', 'last', or False
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return Trueimport pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to input CSV file.
        fill_strategy (str): Strategy for filling missing values.
                             Options: 'mean', 'median', 'mode', 'zero', 'drop'.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values per column:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count}")
            
            if fill_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Removed rows with missing values. New shape: {df_cleaned.shape}")
            else:
                df_cleaned = df.copy()
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df_cleaned[col].isnull().any():
                        if fill_strategy == 'mean':
                            fill_value = df_cleaned[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df_cleaned[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df_cleaned[col].mode()[0]
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            raise ValueError(f"Unknown fill strategy: {fill_strategy}")
                        
                        df_cleaned[col].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value:.2f}")
        else:
            df_cleaned = df
            print("No missing values found.")
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df_cleaned
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data('test_data.csv', fill_strategy='median', output_path='cleaned_data.csv')
    
    if cleaned is not None:
        is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
        print(f"Data validation result: {is_valid}")