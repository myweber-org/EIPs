import numpy as np
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
        (lambda d: isinstance(d, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda d: not d.empty, "DataFrame cannot be empty"),
        (lambda d: d.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check, message in required_checks:
        if not check(df):
            raise ValueError(message)
    return True
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to input CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    output_path (str, optional): Path to save cleaned CSV file
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_method == 'mode':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
            elif fill_method == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            print(f"Missing values filled using {fill_method} method")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
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
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print(f"DataFrame validation passed. Shape: {df.shape}")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean', output_path='cleaned_data.csv')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Validation result: {validation_result}")
        print("Cleaned DataFrame:")
        print(cleaned_df)
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a column in the DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = dataframe.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        
        if max_val - min_val == 0:
            df_copy[f'{column}_normalized'] = 0.5
        else:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        
        if std_val == 0:
            df_copy[f'{column}_normalized'] = 0
        else:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize_method='minmax'):
    """
    Comprehensive data cleaning function.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: List of numeric columns to process (defaults to all numeric columns)
        outlier_threshold: IQR threshold for outlier removal
        normalize_method: Normalization method to use
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_threshold)
            cleaned_df = normalize_column(cleaned_df, column, normalize_method)
    
    return cleaned_df

def get_summary_statistics(dataframe):
    """
    Generate summary statistics for numeric columns.
    
    Args:
        dataframe: pandas DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    numeric_df = dataframe.select_dtypes(include=[np.number])
    
    summary = {
        'mean': numeric_df.mean().to_dict(),
        'median': numeric_df.median().to_dict(),
        'std': numeric_df.std().to_dict(),
        'min': numeric_df.min().to_dict(),
        'max': numeric_df.max().to_dict(),
        'count': numeric_df.count().to_dict()
    }
    
    return summary
import pandas as pd
import re

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Fill numeric nulls with column mean
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # Fill text nulls with 'Unknown'
    text_cols = df_clean.select_dtypes(include=['object']).columns
    for col in text_cols:
        df_clean[col].fillna('Unknown', inplace=True)
    
    # Standardize text: lowercase and remove extra whitespace
    for col in text_cols:
        df_clean[col] = df_clean[col].astype(str).str.lower()
        df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    # Remove duplicate rows
    df_clean.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df):
    """
    Validate that DataFrame meets basic quality standards.
    """
    checks = {
        'has_nulls': df.isnull().sum().sum() == 0,
        'has_duplicates': not df.duplicated().any(),
        'row_count': len(df) > 0,
        'column_count': len(df.columns) > 0
    }
    
    return all(checks.values()), checks

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', None, 'Charlie', 'Alice'],
        'age': [25, None, 30, 35, 25],
        'email': ['ALICE@test.com', 'bob@test.com', 'charlie@test.com', None, 'alice@test.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nNull counts:")
    print(df.isnull().sum())
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, validation_results = validate_dataframe(cleaned_df)
    print(f"\nData validation passed: {is_valid}")
    print(f"Validation details: {validation_results}")import pandas as pd
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
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore' and self.numeric_columns.any():
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            outlier_mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[outlier_mask]
        elif method == 'iqr' and self.numeric_columns.any():
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def get_cleaned_data(self):
        return self.df

def clean_dataset(df, missing_strategy='mean', outlier_method='zscore'):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    cleaner.remove_outliers(method=outlier_method)
    return cleaner.get_cleaned_data()