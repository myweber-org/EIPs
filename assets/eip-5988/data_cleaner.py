import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def clean_dataset_with_threshold(df, null_threshold=0.5):
    """
    Clean DataFrame with configurable null threshold for column removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        null_threshold (float): Threshold for null percentage to drop columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Calculate null percentage for each column
    null_percentage = df.isnull().sum() / len(df)
    
    # Drop columns with null percentage above threshold
    columns_to_drop = null_percentage[null_percentage > null_threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # Fill remaining null values with column mean for numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # Fill categorical columns with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown')
    
    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned.reset_index(drop=True)

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 5],
        'B': [5, 6, 7, None, 9],
        'C': ['x', 'y', 'z', 'x', 'y'],
        'D': [10, 20, 30, 40, 50]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)