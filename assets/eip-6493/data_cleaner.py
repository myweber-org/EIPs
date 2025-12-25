
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for column in columns:
        if column not in data_copy.columns:
            continue
            
        if data_copy[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[column].mean()
            elif strategy == 'median':
                fill_value = data_copy[column].median()
            elif strategy == 'mode':
                fill_value = data_copy[column].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[column] = data_copy[column].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            if normalize_method == 'minmax':
                cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
            elif normalize_method == 'zscore':
                cleaned_data[f"{column}_standardized"] = normalize_zscore(cleaned_data, column)
    
    return cleaned_dataimport pandas as pd
import numpy as np

def load_and_clean_csv(filepath, drop_na=True, fill_missing=False, fill_value=0):
    """
    Load a CSV file and perform basic cleaning operations.
    
    Args:
        filepath (str): Path to the CSV file.
        drop_na (bool): Whether to drop rows with missing values.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if drop_na and fill_missing:
        raise ValueError("Cannot both drop and fill missing values. Choose one.")
    
    if drop_na:
        df = df.dropna()
    elif fill_missing:
        df = df.fillna(fill_value)
    
    return df

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list): Columns to consider for duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df, column_name):
    """
    Normalize a column to range [0, 1].
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    
    col_min = df[column_name].min()
    col_max = df[column_name].max()
    
    if col_max == col_min:
        df[column_name] = 0.5
    else:
        df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    
    return df

def filter_outliers(df, column_name, method='iqr', threshold=1.5):
    """
    Filter outliers from a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to filter.
        method (str): 'iqr' for interquartile range or 'zscore' for standard deviation.
        threshold (float): Threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    
    if method == 'iqr':
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    elif method == 'zscore':
        mean = df[column_name].mean()
        std = df[column_name].std()
        z_scores = np.abs((df[column_name] - mean) / std)
        filtered_df = df[z_scores <= threshold]
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'.")
    
    return filtered_df
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, applies to all object dtype columns.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str)
            
            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].str.upper()
            
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    valid_count = df['email_valid'].sum()
    print(f"Found {valid_count} valid email addresses out of {len(df)} rows.")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE BROWN', '  Bob White  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.co', 'bob@domain.net'],
        'age': [25, 30, 25, 35, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df, columns_to_clean=['name'], case_normalization='lower')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validated = validate_email_column(cleaned, 'email')
    print("DataFrame with email validation:")
    print(validated)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (bool): Whether to fill missing values. Default is True.
    fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, None, 20, 30, None],
        'category': ['A', 'B', 'B', 'C', 'D']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the data
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['id', 'value'])
    print(f"\nValidation: {is_valid}, Message: {message}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fillna_strategy (str): Strategy for filling NaN values ('mean', 'median', 'mode', or 'zero').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fillna_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fillna_strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif fillna_strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                elif fillna_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unsupported fillna_strategy: {fillna_strategy}")
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled NaN in '{col}' with {fillna_strategy} value: {fill_value:.2f}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, None],
        'category': ['A', 'B', 'B', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fillna_strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    print("\nCleaned validation results:")
    print(validate_dataset(cleaned))import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate basic DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including basic statistics.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary containing summary information
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return summary
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): Column index or name to process.
    
    Returns:
    np.ndarray: Data with outliers removed.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if isinstance(column, str):
        # If column is a string, assume data is structured with column names
        # This is a placeholder for actual column indexing logic
        col_data = data
    else:
        col_data = data[:, column]
    
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.ndarray): Input data.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if len(data) == 0:
        return {"mean": None, "median": None, "std": None}
    
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data)
    }

if __name__ == "__main__":
    sample_data = np.random.randn(1000, 3)
    sample_data[50, 1] = 100  # Add an outlier
    
    cleaned_data = remove_outliers_iqr(sample_data, column=1)
    stats = calculate_statistics(cleaned_data[:, 1])
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Statistics for cleaned column: {stats}")import pandas as pd

def remove_duplicates(input_file, output_file, subset=None):
    """
    Load data from a CSV file, remove duplicate rows,
    and save the cleaned data to a new file.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the cleaned CSV file.
    subset (list, optional): Column names to consider for identifying duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        if subset:
            df_cleaned = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_cleaned = df.drop_duplicates(keep='first')
        
        final_count = len(df_cleaned)
        removed_count = initial_count - final_count
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed successfully.")
        print(f"Initial rows: {initial_count}")
        print(f"Final rows: {final_count}")
        print(f"Duplicate rows removed: {removed_count}")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    remove_duplicates('raw_data.csv', 'cleaned_data.csv')
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
    
    return df

def validate_dataframe(df, required_columns):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
    
    return Trueimport pandas as pd
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
            if col in self.df.columns:
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
            if col in self.df.columns:
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
            if col in self.df.columns:
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
            if col in self.df.columns and self.df[col].isnull().any():
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
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[5, 'feature_b'] = 1000
    
    cleaner = DataCleaner(df)
    print(f"Original data shape: {cleaner.original_shape}")
    
    outliers_removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {outliers_removed} outliers")
    
    cleaner.fill_missing_median()
    cleaner.standardize_zscore()
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())