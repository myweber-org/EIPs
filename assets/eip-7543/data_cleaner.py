
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_dfimport numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    numeric_cols = ['feature_a', 'feature_b']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(result.head())import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: Optional[str] = None,
    missing_strategy: str = "mean",
    drop_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing low-quality columns.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Optional path to save cleaned data
    missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    drop_threshold: Maximum fraction of missing values allowed per column
    
    Returns:
    Cleaned DataFrame
    """
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Remove columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Handle remaining missing values
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            if missing_strategy == "mean":
                fill_value = df[column].mean()
            elif missing_strategy == "median":
                fill_value = df[column].median()
            elif missing_strategy == "mode":
                fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
            else:
                continue
            df[column] = df[column].fillna(fill_value)
        else:
            # For categorical columns, fill with most frequent value
            if not df[column].mode().empty:
                df[column] = df[column].fillna(df[column].mode()[0])
    
    # Save cleaned data if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame meets basic quality criteria.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if data passes validation
    """
    if df.empty:
        return False
    
    # Check for remaining missing values
    if df.isnull().sum().sum() > 0:
        return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    cleaned_data = clean_csv_data(
        input_path="raw_data.csv",
        output_path="cleaned_data.csv",
        missing_strategy="median",
        drop_threshold=0.3
    )
    
    if validate_dataframe(cleaned_data):
        print("Data cleaning completed successfully")
        print(f"Cleaned shape: {cleaned_data.shape}")
    else:
        print("Data validation failed")