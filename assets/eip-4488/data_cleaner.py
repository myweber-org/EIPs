
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'original_count': len(df),
        'cleaned_count': len(remove_outliers_iqr(df, column)),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def process_numeric_columns(df, threshold=0.5):
    """
    Automatically process all numeric columns with significant outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): IQR threshold multiplier (default: 1.5)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = df.copy()
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        
        if outlier_count > 0:
            print(f"Processing column '{col}': Found {outlier_count} outliers")
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & 
                                   (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df.reset_index(drop=True)import pandas as pd

def clean_dataset(df):
    """
    Cleans a pandas DataFrame by removing duplicate rows and
    filling missing numeric values with the column mean.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values in numeric columns with the column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    return df_cleaned

def main():
    # Example usage
    data = {
        'A': [1, 2, 2, None, 5],
        'B': [None, 2, 2, 4, 5],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    main()