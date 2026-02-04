import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Error cleaning column {column}: {e}")
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_rows = initial_rows - df_clean.shape[0]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Log cleaning results
    print(f"Removed {removed_rows} duplicate rows.")
    print(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    """
    df_filled = df.copy()
    
    # Select numeric columns
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    elif strategy == 'median':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].median())
    elif strategy == 'zero':
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)
    
    missing_count = df_filled[numeric_cols].isnull().sum().sum()
    print(f"Missing values after filling: {missing_count}")
    
    return df_filled

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Age': [25, 30, 25, 35, np.nan],
        'Score': [85, 90, 85, 95, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\n")
    
    filled_df = handle_missing_values(cleaned_df, strategy='mean')
    print("\nDataset after handling missing values:")
    print(filled_df)import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows,
                       'fill' to fill with column mean (numeric) or mode (categorical).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill':
        for column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
            else:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown')
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
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