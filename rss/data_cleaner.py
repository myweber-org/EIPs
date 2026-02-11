
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', 'zero'.")
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_strategy}: {fill_value}")
    
    return cleaned_df

def validate_dataset(df, check_duplicates=True, check_missing=True):
    """
    Validate a DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        check_duplicates (bool): Check for duplicate rows.
        check_missing (bool): Check for missing values.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    if check_missing:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].to_dict()
        validation_results['missing_values'] = missing_cols
        validation_results['total_missing'] = missing_counts.sum()
    
    return validation_results

def load_and_clean_csv(filepath, **kwargs):
    """
    Load a CSV file and clean it using the clean_dataset function.
    
    Args:
        filepath (str): Path to the CSV file.
        **kwargs: Additional arguments to pass to clean_dataset.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        print(f"Initial shape: {df.shape}")
        
        validation = validate_dataset(df)
        print("Initial validation results:", validation)
        
        cleaned_df = clean_dataset(df, **kwargs)
        print(f"Cleaned shape: {cleaned_df.shape}")
        
        final_validation = validate_dataset(cleaned_df)
        print("Final validation results:", final_validation)
        
        return cleaned_df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 15.7, 10.5, 10.5, np.nan],
        'category': ['A', 'B', 'A', 'C', 'A', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample dataset created.")
    
    cleaned = clean_dataset(df, fill_strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned)