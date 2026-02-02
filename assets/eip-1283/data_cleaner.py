import pandas as pd
import argparse
import sys

def remove_duplicates(input_file, output_file, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        subset (list, optional): Columns to consider for duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False to drop all
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        duplicates_removed = initial_count - final_count
        print(f"Successfully processed {input_file}")
        print(f"Initial rows: {initial_count}")
        print(f"Final rows: {final_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Cleaned data saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate rows from CSV files')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--subset', nargs='+', help='Columns to consider for duplicates')
    parser.add_argument('--keep', choices=['first', 'last', 'none'], 
                       default='first', help='Which duplicate to keep')
    
    args = parser.parse_args()
    
    keep_value = False if args.keep == 'none' else args.keep
    remove_duplicates(args.input, args.output, args.subset, keep_value)

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning function
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
            removal_stats[col] = removed
            
            # Standardize the column
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
            
            # Normalize the column
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df, removal_stats

def validate_data(df, required_columns, min_rows=1):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values")
    
    return True

def generate_summary(df):
    """
    Generate statistical summary of the dataset
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if summary['numeric_columns']:
        numeric_stats = df[summary['numeric_columns']].describe().to_dict()
        summary['numeric_statistics'] = numeric_stats
    
    return summary
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (bool): Whether to fill missing values. Default is False.
    fill_value: Value to use for filling missing data. Default is 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): Method for outlier detection ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df_copy[column].dropna())
        abs_z_scores = abs(z_scores)
        mask = abs_z_scores < threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df_copy[mask]