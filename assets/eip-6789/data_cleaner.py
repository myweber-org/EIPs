import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to apply cleaning to, if None applies to all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    else:
        for col in columns:
            if col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = df_clean[col].mean()
                    elif strategy == 'median':
                        fill_value = df_clean[col].median()
                    elif strategy == 'mode':
                        fill_value = df_clean[col].mode()[0]
                    else:
                        raise ValueError(f"Unsupported strategy: {strategy}")
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    from sklearn.preprocessing import StandardScaler
    
    df_standardized = df.copy()
    
    if columns is None:
        numeric_cols = df_standardized.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    scaler = StandardScaler()
    df_standardized[columns] = scaler.fit_transform(df_standardized[columns])
    
    return df_standardized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Args:
        filepath: Path to the CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns_to_drop: List of column names to drop (optional)
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isnull().any():
                if missing_strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif missing_strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif missing_strategy == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif missing_strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
        
        for column in df.select_dtypes(include=['object']).columns:
            if df[column].isnull().any():
                df[column].fillna('Unknown', inplace=True)
        
        df = df.reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path for output CSV file
    """
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data exported to {output_path}")
    else:
        print("No data to export")
import numpy as np
import pandas as pd

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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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
    Complete data processing pipeline: remove outliers and calculate statistics.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    tuple: (cleaned_df, stats_dict)
    """
    cleaned_df = remove_outliers_iqr(df, column)
    stats = calculate_basic_stats(cleaned_df, column)
    
    return cleaned_df, stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df, stats = process_dataframe(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    print("\nStatistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
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
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_df, 'values'))

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    handling missing values in numeric columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values in numeric columns with column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that DataFrame has no missing values in numeric columns
    and no duplicate rows.
    """
    # Check for duplicates
    has_duplicates = df.duplicated().any()
    
    # Check for missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    has_missing = df[numeric_cols].isnull().any().any()
    
    return not has_duplicates and not has_missing

def process_file(input_path, output_path=None):
    """
    Read a CSV file, clean the data, and optionally save to output path.
    """
    # Read input file
    df = pd.read_csv(input_path)
    
    # Clean the data
    df_cleaned = clean_dataframe(df)
    
    # Validate the cleaning
    if validate_dataframe(df_cleaned):
        print("Data cleaning successful. No duplicates or missing values found.")
    else:
        print("Warning: Data cleaning may not have completed successfully.")
    
    # Save to output if specified
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df_cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, 2, 4, np.nan],
        'B': [5, 6, 6, np.nan, 9],
        'C': ['x', 'y', 'y', 'z', 'w']
    })
    
    print("Original DataFrame:")
    print(sample_data)
    
    cleaned = clean_dataframe(sample_data)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned)
    print(f"\nData validation passed: {is_valid}")