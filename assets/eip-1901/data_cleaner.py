
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
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
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.3, 20.1, None],
        'category': ['A', 'B', 'B', None, 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: Specific columns to fill, fills all if None
    
    Returns:
        DataFrame with filled missing values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df_filled.columns
    
    for col in columns:
        if df_filled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize, normalizes all numeric if None
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'minmax':
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val > 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return df_normalized

def clean_dataframe(df: pd.DataFrame, 
                    remove_dups: bool = True,
                    fill_na: bool = True,
                    fill_strategy: str = 'mean',
                    normalize: bool = False,
                    norm_method: str = 'minmax') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicate rows
        fill_na: Whether to fill missing values
        fill_strategy: Strategy for filling missing values
        normalize: Whether to normalize numeric columns
        norm_method: Normalization method
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicate_rows(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy=fill_strategy)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df, method=norm_method)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str],
                       min_rows: int = 1) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: Columns that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'issues': []
    }
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        validation_result['missing_columns'] = missing_cols
        validation_result['is_valid'] = False
        validation_result['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check minimum rows
    if len(df) < min_rows:
        validation_result['is_valid'] = False
        validation_result['issues'].append(f"DataFrame has only {len(df)} rows, minimum required is {min_rows}")
    
    # Check for empty columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        validation_result['empty_columns'] = empty_cols
        validation_result['issues'].append(f"Empty columns detected: {empty_cols}")
    
    return validation_result
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

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            df = remove_outliers_iqr(df, col)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        print(f"Original rows: {len(pd.read_csv(file_path))}, Cleaned rows: {len(df)}")
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)