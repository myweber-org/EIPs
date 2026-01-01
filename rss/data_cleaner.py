
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[1000:1005, 'A'] = 500
    sample_data.loc[1006:1010, 'B'] = 1000
    
    print("Original shape:", sample_data.shape)
    cleaned_data = clean_dataset(sample_data)
    print("Cleaned shape:", cleaned_data.shape)
    print("\nSummary statistics:")
    print(cleaned_data.describe())
import pandas as pd

def clean_dataset(df, id_column=None):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        id_column (str, optional): Column name to use for identifying duplicates.
                                   If None, checks all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Standardize column names: lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    if id_column and id_column in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=[id_column])
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'User ID': [1, 2, 2, 3, 4],
#         'First Name': ['John', 'Jane', 'Jane', 'Bob', 'Alice'],
#         'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Brown'],
#         'Email': ['john@example.com', 'jane@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     cleaned = clean_dataset(df, id_column='User ID')
#     print(cleaned)
#     
#     # Validate
#     is_valid, message = validate_dataframe(cleaned, required_columns=['user_id', 'email'])
#     print(f"\nValidation: {is_valid} - {message}")