
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def main():
    data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10, 9, 8, 15, 200]}
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame (outliers removed):")
    print(cleaned_df)
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")

if __name__ == "__main__":
    main()