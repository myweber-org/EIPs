
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean dataset by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().any():
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    return df_cleaned

def validate_data(df):
    """
    Validate data after cleaning.
    """
    # Check for remaining missing values
    missing_values = df.isnull().sum().sum()
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    
    validation_report = {
        'missing_values': missing_values,
        'duplicates': duplicates,
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }
    
    return validation_report

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'A': [1, 2, 2, 3, None],
        'B': ['x', 'y', 'y', None, 'z'],
        'C': [1.1, 2.2, 2.2, 3.3, 4.4]
    })
    
    print("Original data:")
    print(data)
    
    cleaned_data = clean_dataset(data)
    print("\nCleaned data:")
    print(cleaned_data)
    
    report = validate_data(cleaned_data)
    print("\nValidation report:")
    for key, value in report.items():
        print(f"{key}: {value}")