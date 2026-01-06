
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, missing_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input pandas DataFrame
        drop_duplicates: Boolean flag to remove duplicate rows
        fill_missing: Boolean flag to fill missing values
        missing_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().sum() > 0:
                if missing_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif missing_strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif missing_strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                elif missing_strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[col].mean()
                
                missing_count = cleaned_df[col].isnull().sum()
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{col}' with {missing_strategy}: {fill_value}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

def main():
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', np.nan, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing=True, missing_strategy='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nDataFrame validation: {is_valid}")

if __name__ == "__main__":
    main()