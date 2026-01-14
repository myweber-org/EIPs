import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    return summary

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary:")
    print(get_data_summary(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nDataFrame validation: {is_valid}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        print(f"Total missing values: {missing_count}")
        
        if missing_count > 0:
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Duplicate rows removed: {duplicates_removed}")
        
        # Save cleaned data
        if output_path is None:
            output_path = file_path.replace('.csv', '_cleaned.csv')
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Cleaned data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate the cleaned dataframe for common issues.
    """
    if df is None:
        print("DataFrame is None")
        return False
    
    validation_passed = True
    
    # Check for remaining missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame still contains missing values")
        validation_passed = False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Warning: Column {col} contains infinite values")
            validation_passed = False
    
    # Check for empty strings in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if (df[col] == '').any():
            print(f"Warning: Column {col} contains empty strings")
            validation_passed = False
    
    if validation_passed:
        print("Data validation passed")
    
    return validation_passed

if __name__ == "__main__":
    # Example usage
    input_file = "sample_data.csv"
    cleaned_df = clean_csv_data(input_file)
    
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)