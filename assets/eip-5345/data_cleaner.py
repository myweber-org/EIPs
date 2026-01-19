
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if normalize_text and columns_to_clean:
        for col in columns_to_clean:
            if col in df_clean.columns and df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].apply(_normalize_string)
                print(f"Normalized text in column: {col}")
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string: lowercase, remove extra whitespace, and strip.
    """
    if pd.isna(text):
        return text
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['is_valid_email'] = df[email_column].apply(lambda x: bool(re.match(pattern, str(x))) if pd.notna(x) else False)
    
    valid_count = df['is_valid_email'].sum()
    total_count = len(df)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)")
    
    return dfimport pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns, method='zscore'):
    """
    Normalize specified columns in a DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
        method: normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = stats.zscore(df[col])
        elif method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        elif method == 'robust':
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            if iqr != 0:
                df_normalized[col] = (df[col] - median) / iqr
    
    return df_normalized

def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        method: outlier detection method ('iqr', 'zscore')
        threshold: threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            mask = z_scores < threshold
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (None for all columns)
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            df_processed[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df_processed[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_processed[col] = df[col].fillna(df[col].mode()[0])
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_columns: list of columns that should be numeric
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if numeric_columns:
        non_numeric_cols = [col for col in numeric_columns if col in df.columns 
                           and not np.issubdtype(df[col].dtype, np.number)]
        if non_numeric_cols:
            return False, f"Non-numeric columns: {non_numeric_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def create_data_summary(df):
    """
    Create a summary of the DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary containing data summary
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return summary
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {col}")
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")
        print(f"Final dataset shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                       Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0]
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 4],
        'B': [5, None, 5, 6, 7, 8],
        'C': ['x', 'y', 'x', 'z', 'y', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning data...")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")