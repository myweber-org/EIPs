import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled. Default is None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics: {calculate_summary_statistics(df, 'values')}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics: {calculate_summary_statistics(cleaned_df, 'values')}")

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Clean CSV data by handling missing values, converting data types,
    and removing duplicates.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_patterns = ['date', 'time', 'timestamp', 'created', 'updated']
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    continue
        
        # Remove rows where all values are NaN
        df = df.dropna(how='all')
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Generate cleaning report
        report = {
            'original_rows': initial_count,
            'cleaned_rows': len(df),
            'duplicates_removed': duplicates_removed,
            'columns_processed': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'output_path': output_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
        
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    validation_results = {}
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    validation_results['total_missing_values'] = int(missing_values)
    
    # Check data types
    validation_results['dtypes'] = df.dtypes.to_dict()
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    infinite_count = np.isinf(numeric_df).sum().sum()
    validation_results['infinite_values'] = int(infinite_count)
    
    # Check for zeros in numeric columns (potential issue)
    zero_count = (numeric_df == 0).sum().sum()
    validation_results['zero_values'] = int(zero_count)
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    result = clean_csv_data(input_file, output_file)
    
    if result:
        print("Data cleaning completed successfully!")
        print(f"Report: {result}")
        
        # Load and validate cleaned data
        cleaned_df = pd.read_csv(output_file)
        validation = validate_dataframe(cleaned_df)
        print(f"Validation results: {validation}")
    else:
        print("Data cleaning failed!")