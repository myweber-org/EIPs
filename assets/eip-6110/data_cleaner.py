
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    sample_data = {'values': [1, 2, 3, 4, 5, 100, 200, 300]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after outlier removal:")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_statistics(df, 'values')
    print("Summary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("DataFrame after min-max normalization:")
    print(normalized_df)
import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8, missing_strategy='median'):
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        duplicate_threshold: threshold for considering rows as duplicates (0.0 to 1.0)
        missing_strategy: strategy for handling missing values ('median', 'mean', 'drop', 'fill')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Remove approximate duplicates based on threshold
    if duplicate_threshold < 1.0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            similarity_matrix = df[numeric_cols].corr().abs()
            duplicate_mask = similarity_matrix > duplicate_threshold
            np.fill_diagonal(duplicate_mask.values, False)
            duplicate_pairs = np.where(duplicate_mask)
            
            if len(duplicate_pairs[0]) > 0:
                duplicate_indices = set()
                for i, j in zip(duplicate_pairs[0], duplicate_pairs[1]):
                    if i < j:
                        duplicate_indices.add(j)
                
                df = df.drop(index=list(duplicate_indices)).reset_index(drop=True)
    
    # Handle missing values
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['median', 'mean']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if missing_strategy == 'median':
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
    elif missing_strategy == 'fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    
    # Log cleaning results
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]}")
    print(f"Columns removed: {original_shape[1] - df.shape[1]}")
    
    return df

def validate_dataset(df, min_rows=10, required_columns=None):
    """
    Validate dataset meets minimum requirements.
    
    Args:
        df: pandas DataFrame to validate
        min_rows: minimum number of rows required
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if dataset is valid
    """
    if df.shape[0] < min_rows:
        print(f"Dataset has only {df.shape[0]} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Column '{col}' contains infinite values")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'feature_a': [1, 2, 2, 3, 4, 5, None, 7, 8, 9],
        'feature_b': [10, 20, 20, 30, 40, 50, 60, 70, 80, 90],
        'feature_c': [100, 200, 200, 300, 400, 500, 600, 700, 800, 900],
        'constant_col': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(sample_data)
    cleaned_df = clean_dataset(df, duplicate_threshold=0.95, missing_strategy='median')
    
    is_valid = validate_dataset(cleaned_df, min_rows=5, required_columns=['feature_a', 'feature_b'])
    print(f"Dataset is valid: {is_valid}")import pandas as pd
import numpy as np

def clean_csv_data(filepath, drop_na=True, fill_strategy='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        drop_na (bool): If True, drop rows with missing values.
        fill_strategy (str): Strategy to fill missing values if drop_na is False.
                             Options: 'mean', 'median', 'mode', or 'zero'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    
    if df.empty:
        print("Warning: The loaded DataFrame is empty.")
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if drop_na:
        df_cleaned = df.dropna()
    else:
        df_cleaned = df.copy()
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif fill_strategy == 'median':
                    fill_value = df_cleaned[col].median()
                elif fill_strategy == 'mode':
                    fill_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 0
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = df_cleaned[col].mean()
                df_cleaned[col].fillna(fill_value, inplace=True)
        
        for col in non_numeric_cols:
            df_cleaned[col].fillna('Unknown', inplace=True)
    
    df_cleaned = df_cleaned.reset_index(drop=True)
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_cleaned.shape}")
    return df_cleaned

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame.")
        return df
    
    if not np.issubdtype(df[column].dtype, np.number):
        print(f"Error: Column '{column}' is not numeric.")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    outliers_removed = len(df) - len(df_filtered)
    print(f"Removed {outliers_removed} outliers from column '{column}'.")
    
    return df_filtered.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', None, 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', drop_na=False, fill_strategy='median')
    if cleaned is not None:
        print("Cleaned DataFrame head:")
        print(cleaned.head())
        
        cleaned_no_outliers = remove_outliers_iqr(cleaned, 'A')
        print("DataFrame after outlier removal head:")
        print(cleaned_no_outliers.head())
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_file, output_file=None):
    """
    Load and clean CSV data by handling missing values,
    removing duplicates, and standardizing formats.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        missing_after = df.isnull().sum().sum()
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"cleaned_{input_path.name}"
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        # Print cleaning summary
        print(f"Data cleaning completed successfully!")
        print(f"Original rows: {initial_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Missing values handled: {missing_before} -> {missing_after}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df, output_file
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for any remaining NaN values
    if df.isnull().any().any():
        print("Warning: DataFrame still contains NaN values.")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, None, 30, 35, 28],
        'City': ['New York', 'London', None, 'Paris', 'Tokyo'],
        'Score': [85.5, 92.0, 78.5, None, 88.0]
    }
    
    # Create sample CSV for testing
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df, output_path = clean_csv_data('sample_data.csv')
    
    if cleaned_df is not None:
        print("\nCleaned DataFrame preview:")
        print(cleaned_df.head())
        
        # Validate the cleaned data
        is_valid = validate_dataframe(cleaned_df, ['name', 'age', 'city', 'score'])
        print(f"\nData validation: {'PASSED' if is_valid else 'FAILED'}")
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
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: Column name to analyze
    
    Returns:
        Dictionary with statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df: pandas DataFrame
        columns: List of column names or None
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                continue
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_basic_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_basic_stats(cleaned_df, 'A'))