
import pandas as pd
import re

def clean_dataframe(df, text_column='text', id_column='id'):
    """
    Clean a DataFrame by removing duplicate rows and normalizing text in a specified column.
    """
    # Remove duplicate rows based on the id column
    df_cleaned = df.drop_duplicates(subset=[id_column], keep='first')
    
    # Normalize text: lowercase and remove extra whitespace
    if text_column in df_cleaned.columns:
        df_cleaned[text_column] = df_cleaned[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
        )
    
    return df_cleaned

def load_and_clean_csv(file_path, text_column='text', id_column='id'):
    """
    Load a CSV file and clean the data.
    """
    try:
        df = pd.read_csv(file_path)
        cleaned_df = clean_dataframe(df, text_column, id_column)
        return cleaned_df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_data = load_and_clean_csv(input_file)
    
    if cleaned_data is not None:
        cleaned_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(cleaned_data)}")import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = dataframe.columns.tolist()
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    print(f"Removed {len(dataframe) - len(cleaned_df)} duplicate rows")
    print(f"Original shape: {dataframe.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['object']).columns
    
    cleaned_df = dataframe.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 1, 2, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'score': ['85', '90', '78', '85', '90', '92']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'score'])
    print(f"\nDataFrame validation: {is_valid}")
    
    numeric_df = clean_numeric_columns(cleaned_df, columns=['score'])
    print("\nDataFrame with cleaned numeric columns:")
    print(numeric_df)
    print(f"\nScore column dtype: {numeric_df['score'].dtype}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        numeric_columns: List of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
    
    cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def calculate_statistics(dataframe, columns=None):
    """
    Calculate basic statistics for specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to analyze
    
    Returns:
        Dictionary with statistics
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = {}
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            stats[col] = {
                'mean': dataframe[col].mean(),
                'median': dataframe[col].median(),
                'std': dataframe[col].std(),
                'min': dataframe[col].min(),
                'max': dataframe[col].max(),
                'count': dataframe[col].count()
            }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics:")
    stats = calculate_statistics(df)
    for col, col_stats in stats.items():
        print(f"{col}: {col_stats}")
    
    cleaned_df = clean_dataset(df, ['feature_a', 'feature_b'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    cleaned_stats = calculate_statistics(cleaned_df)
    for col, col_stats in cleaned_stats.items():
        print(f"{col}: {col_stats}")