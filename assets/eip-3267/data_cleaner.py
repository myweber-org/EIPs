
import pandas as pd

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize string columns: strip whitespace and convert to lowercase
    for col in column_names:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using a simple regex pattern.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    # Create a mask for valid emails
    valid_mask = df[email_column].astype(str).str.match(pattern)
    
    # Return DataFrame with valid emails and validation status column
    result_df = df.copy()
    result_df['email_valid'] = valid_mask
    
    return result_df[valid_mask], result_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(dataframe, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': dataframe[column].count()
    }
    
    return stats

def clean_dataset(dataframe, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    # Add some outliers
    df.loc[1000] = [500, 1000, 300]  # Outlier row
    
    cleaned_df = clean_dataset(df)
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    for col in ['A', 'B', 'C']:
        stats = calculate_basic_stats(cleaned_df, col)
        print(f"\nStatistics for {col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")