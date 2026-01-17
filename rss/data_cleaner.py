
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Clean CSV data by handling missing values and converting data types.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Display initial info
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing numeric values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Convert date columns if present
        date_patterns = ['date', 'time', 'created', 'updated']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Converted {col} to datetime")
                except:
                    print(f"Could not convert {col} to datetime")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        print(f"Final shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None:
        return False
    
    checks = {
        'has_data': len(df) > 0,
        'has_columns': len(df.columns) > 0,
        'no_null_values': df.isnull().sum().sum() == 0,
        'no_infinite_values': np.isfinite(df.select_dtypes(include=[np.number])).all().all()
    }
    
    print("Data Validation Results:")
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  {check_name}: {status}")
    
    return all(checks.values())

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validation_passed = validate_dataframe(cleaned_df)
        if validation_passed:
            print("Data cleaning completed successfully")
        else:
            print("Data cleaning completed with warnings")