
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Clean CSV data by handling missing values and converting data types.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Display initial info
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing numeric values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Convert date columns if present
        date_patterns = ['date', 'time', 'created', 'updated']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Converted {col} to datetime")
                except:
                    print(f"Could not convert {col} to datetime")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        print(f"Final shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None:
        return False
    
    checks = {
        'has_data': len(df) > 0,
        'has_columns': len(df.columns) > 0,
        'no_null_values': df.isnull().sum().sum() == 0,
        'no_infinite_values': np.isfinite(df.select_dtypes(include=[np.number])).all().all()
    }
    
    print("Data Validation Results:")
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  {check_name}: {status}")
    
    return all(checks.values())

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validation_passed = validate_dataframe(cleaned_df)
        if validation_passed:
            print("Data cleaning completed successfully")
        else:
            print("Data cleaning completed with warnings")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def main():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    stats = calculate_summary_statistics(df, 'value')
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    removed_count = len(df) - len(cleaned_df)
    print(f"\nRemoved {removed_count} outliers")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import sys

def clean_csv_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        df = df.drop_duplicates()
        
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Data saved to: {output_file}")
        print("Data cleaning completed successfully.")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv_data(input_file, output_file)
    
    if not success:
        sys.exit(1)import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def clean_dataset_with_threshold(df, null_threshold=0.5):
    """
    Clean DataFrame with configurable null value threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        null_threshold (float): Threshold for column null percentage (0-1).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Calculate null percentage for each column
    null_percentage = df.isnull().sum() / len(df)
    
    # Identify columns with null percentage below threshold
    columns_to_keep = null_percentage[null_percentage <= null_threshold].index
    
    # Keep only columns below threshold
    df_filtered = df[columns_to_keep]
    
    # Remove rows with null values in remaining columns
    df_cleaned = df_filtered.dropna()
    
    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': [10, 20, 30, None, 50, 50],
        'C': [100, 200, 300, 400, 500, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)