
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: Method for imputation ('mean', 'median', 'mode', 'drop')
        columns: List of columns to process, None processes all columns
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: List of numeric columns to check for outliers
        threshold: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        mean = df_standardized[col].mean()
        std = df_standardized[col].std()
        
        if std > 0:
            df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of columns that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np
from pathlib import Path

def load_csv_data(file_path):
    """Load CSV file into pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path} with {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_missing_values(df, strategy='mean', columns=None):
    """Handle missing values in DataFrame."""
    if df is None or df.empty:
        return df
    
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.columns
    
    for col in columns:
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                missing_count = df_cleaned[col].isnull().sum()
                print(f"Column '{col}' has {missing_count} missing values")
                
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    fill_value = df_cleaned[col].mean()
                    df_cleaned[col].fillna(fill_value, inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    fill_value = df_cleaned[col].median()
                    df_cleaned[col].fillna(fill_value, inplace=True)
                elif strategy == 'mode':
                    fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else None
                    df_cleaned[col].fillna(fill_value, inplace=True)
                elif strategy == 'drop':
                    df_cleaned = df_cleaned.dropna(subset=[col])
                else:
                    df_cleaned[col].fillna(0, inplace=True)
    
    return df_cleaned

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    if df is None or df.empty:
        return df
    
    initial_rows = len(df)
    df_deduped = df.drop_duplicates(subset=subset, keep='first')
    removed_count = initial_rows - len(df_deduped)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return df_deduped

def normalize_numeric_columns(df, columns=None):
    """Normalize numeric columns to range [0, 1]."""
    if df is None or df.empty:
        return df
    
    df_normalized = df.copy()
    
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    for col in columns:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            
            if col_max > col_min:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
                print(f"Normalized column '{col}' to range [0, 1]")
    
    return df_normalized

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV file."""
    if df is None or df.empty:
        print("No data to save")
        return False
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def process_data_pipeline(input_file, output_file, cleaning_strategy='mean'):
    """Complete data cleaning pipeline."""
    print(f"Starting data cleaning pipeline for {input_file}")
    
    df = load_csv_data(input_file)
    if df is None:
        return False
    
    print("Step 1: Handling missing values...")
    df = clean_missing_values(df, strategy=cleaning_strategy)
    
    print("Step 2: Removing duplicates...")
    df = remove_duplicates(df)
    
    print("Step 3: Normalizing numeric columns...")
    df = normalize_numeric_columns(df)
    
    print(f"Final dataset: {len(df)} rows and {len(df.columns)} columns")
    
    success = save_cleaned_data(df, output_file)
    
    if success:
        print("Data cleaning pipeline completed successfully")
    else:
        print("Data cleaning pipeline failed")
    
    return success

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10.5, 20.3, np.nan, 15.7, 25.1, 30.4, np.nan, 18.9, 22.6, 28.3],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'score': [85, 92, 78, 88, 95, 91, 82, 89, 94, 87]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = "test_data.csv"
    test_df.to_csv(test_file, index=False)
    
    output_file = "cleaned_data.csv"
    process_data_pipeline(test_file, output_file)
    
    Path(test_file).unlink(missing_ok=True)