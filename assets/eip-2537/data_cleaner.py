import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
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
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
    
    Returns:
        dict: Dictionary containing summary statistics.
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

def process_dataframe(df, column):
    """
    Main function to process DataFrame by removing outliers and calculating statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize a column using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize a column using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (if None, auto-detect)
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            cleaned_data[column + '_normalized'] = normalize_minmax(cleaned_data, column)
            cleaned_data[column + '_standardized'] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_summary_statistics(data):
    """
    Get summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return pd.DataFrame()
    
    summary = numeric_data.describe().T
    summary['skewness'] = numeric_data.skew()
    summary['kurtosis'] = numeric_data.kurtosis()
    summary['missing_values'] = numeric_data.isnull().sum()
    summary['missing_percentage'] = (summary['missing_values'] / len(data)) * 100
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary statistics:")
    print(get_summary_statistics(sample_data))
    
    cleaned = clean_dataset(sample_data)
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned summary statistics:")
    print(get_summary_statistics(cleaned))
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. 
                                     If None, overwrites input file.
        subset (list, optional): Columns to consider for identifying duplicates.
    
    Returns:
        int: Number of duplicates removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_clean = df.drop_duplicates(keep='first')
        
        final_rows = len(df_clean)
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_rows}")
        print(f"Clean rows: {final_rows}")
        print(f"Output saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        return -1
    except Exception as e:
        print(f"Error: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)