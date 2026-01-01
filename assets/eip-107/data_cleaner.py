
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove rows where critical columns are still missing
        critical_cols = [col for col in df.columns if 'id' in col.lower() or 'key' in col.lower()]
        if critical_cols:
            df = df.dropna(subset=critical_cols)
        
        print(f"Final cleaned shape: {df.shape}")
        
        # Save cleaned data
        if output_path is None:
            output_path = Path(input_path).stem + '_cleaned.csv'
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None, None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check for remaining missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        return False, f"DataFrame still contains {missing_count} missing values"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'user_id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    # Create test DataFrame
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the data
    cleaned_df, output_file = clean_csv_data('test_data.csv')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        is_valid, message = validate_dataframe(cleaned_df, ['user_id', 'name'])
        print(f"Validation: {is_valid} - {message}")
        
        # Display cleaned data
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
        # Clean up test file
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        if os.path.exists(output_file):
            os.remove(output_file)