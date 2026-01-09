
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
        
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name to analyze
        
    Returns:
        Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        
    Returns:
        DataFrame with normalized column added as 'column_normalized'
    """
    min_val = df[column].min()
    max_val = df[column].max()
    
    df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns in a DataFrame.
    
    Args:
        df: pandas DataFrame
        numeric_columns: List of column names to process
        
    Returns:
        Processed DataFrame and statistics dictionary
    """
    processed_df = df.copy()
    all_stats = {}
    
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df = remove_outliers_iqr(processed_df, col)
            stats = calculate_statistics(processed_df, col)
            all_stats[col] = stats
            processed_df = normalize_column(processed_df, col)
    
    return processed_df, all_stats

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame shape:", df.shape)
    
    numeric_cols = ['value', 'score']
    processed_df, statistics = process_dataframe(df, numeric_cols)
    
    print("Processed DataFrame shape:", processed_df.shape)
    print("\nStatistics:")
    for col, stats in statistics.items():
        print(f"\n{col}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")