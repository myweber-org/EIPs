import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Clean a CSV file by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop').
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows.")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        if fill_strategy == 'drop':
            df.dropna(inplace=True)
        elif fill_strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
        elif fill_strategy == 'mode':
            for col in df.columns:
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value[0], inplace=True)
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values handled: {missing_before} -> {missing_after}")
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        print(f"Cleaned data shape: {df.shape}")
        
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
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Validation warning: Column '{col}' contains infinite values.")
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 4],
        'B': [5, np.nan, 7, 8, 8],
        'C': ['x', 'y', 'z', 'x', 'x']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Data validation result: {is_valid}")
        print("\nCleaned DataFrame:")
        print(cleaned_df)