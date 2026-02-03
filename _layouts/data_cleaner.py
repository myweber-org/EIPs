import pandas as pd

def clean_dataframe(df):
    """
    Remove duplicate rows and fill missing values with column mean for numeric columns.
    For categorical columns, fill missing values with the most frequent value.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in ['int64', 'float64']:
            # Fill numeric columns with mean
            mean_value = df_cleaned[column].mean()
            df_cleaned[column].fillna(mean_value, inplace=True)
        else:
            # Fill categorical columns with mode
            mode_value = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown'
            df_cleaned[column].fillna(mode_value, inplace=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the dataframe has no missing values after cleaning.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("Data validation passed: No missing values found.")
        return True
    else:
        print(f"Data validation failed: {missing_values} missing values found.")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5.1, None, 7.3, 8.4, 9.5],
        'C': ['apple', 'banana', 'apple', None, 'cherry']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation_result = validate_dataframe(cleaned_df)