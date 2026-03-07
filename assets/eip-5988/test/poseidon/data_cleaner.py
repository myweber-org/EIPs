
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (bool): Whether to fill missing values. Default is True.
    fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    threshold (float): Z-score threshold. Default is 3.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    from scipy import stats
    import numpy as np
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    filtered_entries = z_scores < threshold
    return df[filtered_entries]

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1] using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    df_copy = df.copy()
    min_val = df_copy[column].min()
    max_val = df_copy[column].max()
    
    if max_val > min_val:
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    return df_copy

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, 6, 6, None, 9],
        'C': [10, 11, 11, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{col}'")
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_data = clean_dataset(input_file, output_file)