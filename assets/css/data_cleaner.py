
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

def calculate_summary_statistics(df, column):
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
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df)
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Main function to load, clean, and analyze a dataset.
    
    Args:
        file_path (str): Path to CSV file
        column_to_clean (str): Column name to clean
    
    Returns:
        tuple: Cleaned DataFrame and summary statistics
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        cleaned_df = remove_outliers_iqr(df, column_to_clean)
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print(f"Removed {len(df) - len(cleaned_df)} outliers")
        
        stats = calculate_summary_statistics(cleaned_df, column_to_clean)
        
        return cleaned_df, stats
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None, None
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None, None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 10, 95),
            np.random.normal(200, 10, 5)
        ])
    })
    
    cleaned_data, statistics = process_dataset(sample_data, 'values')
    
    if cleaned_data is not None:
        print("\nSummary Statistics:")
        for key, value in statistics.items():
            print(f"{key}: {value:.2f}")