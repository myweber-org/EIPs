
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows,
    filling missing numeric values with the column median,
    and filling missing categorical values with 'Unknown'.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in [np.float64, np.int64]:
            # Fill numeric columns with median
            median_value = df_cleaned[column].median()
            df_cleaned[column].fillna(median_value, inplace=True)
        else:
            # Fill categorical columns with 'Unknown'
            df_cleaned[column].fillna('Unknown', inplace=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the DataFrame has no missing values after cleaning.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        return True
    else:
        print(f"DataFrame still has {missing_values} missing values.")
        return False

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    data = {
        'id': [1, 2, 3, 1, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve'],
        'age': [25, 30, np.nan, 25, 35],
        'score': [85.5, 90.0, 78.5, 85.5, np.nan],
        'department': ['HR', 'IT', 'IT', 'HR', None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nMissing values in original:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaning
    is_valid = validate_dataframe(cleaned_df)
    print(f"\nData validation passed: {is_valid}")