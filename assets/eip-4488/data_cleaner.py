
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
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column of a dataset using the IQR method.
    
    Parameters:
    data (pandas.DataFrame): The input dataset.
    column (str): The column name to clean.
    
    Returns:
    pandas.DataFrame: The dataset with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a specified column.
    
    Parameters:
    data (pandas.DataFrame): The input dataset.
    column (str): The column name for statistics.
    
    Returns:
    dict: A dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std_dev': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 14, 15, 16, 17, 18, 100]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nCleaned data:")
    print(cleaned_data)
    
    stats = calculate_basic_stats(cleaned_data, 'values')
    print("\nBasic statistics for cleaned data:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    Returns filtered DataFrame.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns filtered DataFrame.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    Returns normalized Series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score normalization.
    Returns normalized Series.
    """
    mean = data[column].mean()
    std = data[column].std()
    return (data[column] - mean) / std

def handle_missing_mean(data, column):
    """
    Fill missing values with column mean.
    Returns Series with filled values.
    """
    return data[column].fillna(data[column].mean())

def handle_missing_median(data, column):
    """
    Fill missing values with column median.
    Returns Series with filled values.
    """
    return data[column].fillna(data[column].median())

def create_sample_data():
    """
    Create sample DataFrame for testing.
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 200, 100)
    }
    df = pd.DataFrame(data)
    df.iloc[5, 0] = np.nan
    df.iloc[10, 1] = np.nan
    df.iloc[15, 2] = np.nan
    return df

if __name__ == "__main__":
    df = create_sample_data()
    print("Original data shape:", df.shape)
    
    df_clean = remove_outliers_iqr(df, 'feature_a')
    print("After IQR outlier removal:", df_clean.shape)
    
    df['feature_a_normalized'] = normalize_minmax(df, 'feature_a')
    df['feature_b_normalized'] = normalize_zscore(df, 'feature_b')
    
    df['feature_a_filled'] = handle_missing_mean(df, 'feature_a')
    df['feature_b_filled'] = handle_missing_median(df, 'feature_b')
    
    print("Data cleaning completed.")
    print(df.head())