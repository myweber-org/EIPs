import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path=None):
    """
    Clean dataset by removing outliers from numerical columns.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str, optional): Path to save cleaned data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df = pd.read_csv(file_path)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        initial_count = len(df)
        df = remove_outliers_iqr(df, col)
        removed_count = initial_count - len(df)
        print(f"Removed {removed_count} outliers from column '{col}'")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_dataset(input_file, output_file)
        print(f"Data cleaning complete. Original shape: {pd.read_csv(input_file).shape}, Cleaned shape: {cleaned_df.shape}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns:
            original_count = len(cleaned_data)
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
            removed_count = original_count - len(cleaned_data)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_data