import pandas as pd

def clean_dataset(df, subset_columns=None, fill_strategy='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset_columns (list, optional): Columns to consider for duplicate removal.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicates
    if subset_columns:
        df_clean = df_clean.drop_duplicates(subset=subset_columns)
    else:
        df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    categorical_cols = df_clean.select_dtypes(exclude=['number']).columns
    
    if fill_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif fill_strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
    elif fill_strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna(df_clean[categorical_cols].mode().iloc[0])
    elif fill_strategy == 'mode':
        for col in df_clean.columns:
            if col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
    else:
        raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")
    
    return df_clean

# Example usage (commented out)
# if __name__ == "__main__":
#     sample_data = {'A': [1, 2, None, 4, 4],
#                    'B': [5, None, 7, 8, 8],
#                    'C': ['x', 'y', 'z', None, 'x']}
#     df = pd.DataFrame(sample_data)
#     cleaned_df = clean_dataset(df, fill_strategy='mean')
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str, optional): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or 'drop'. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
        min_rows (int, optional): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"