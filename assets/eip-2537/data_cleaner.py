
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers from specified columns or all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): List of column names to clean. If None, cleans all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                original_len = len(cleaned_df)
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
                removed_count = original_len - len(cleaned_df)
                print(f"Removed {removed_count} outliers from column '{column}'")
            else:
                print(f"Column '{column}' is not numeric, skipping")
        else:
            print(f"Column '{column}' not found in DataFrame")
    
    return cleaned_df

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'score': np.random.normal(50, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nDescriptive statistics:")
    print(df[['value', 'score']].describe())
    
    cleaned_df = clean_numeric_data(df, columns=['value', 'score'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned descriptive statistics:")
    print(cleaned_df[['value', 'score']].describe())
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()