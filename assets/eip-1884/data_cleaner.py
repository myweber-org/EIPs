
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, remove_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        remove_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with validation results
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_df = df.copy()
    validation_df['email_valid'] = validation_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = validation_df['email_valid'].sum()
    total_count = len(validation_df)
    
    print(f"Email validation results: {valid_count}/{total_count} valid emails ({valid_count/total_count*100:.1f}%)")
    
    return validation_df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the file
        format (str): File format ('csv', 'excel', 'json')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'json'")
    
    print(f"Data saved to {output_path} in {format} format")
import csv
import sys
from pathlib import Path

def remove_duplicates(input_file, output_file=None, key_column=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        key_column: Column name to identify duplicates (optional)
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')
    
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            if key_column:
                key = row.get(key_column)
            else:
                key = tuple(row.values())
            
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    return len(unique_rows)

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file] [key_column]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    key_column = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        count = remove_duplicates(input_file, output_file, key_column)
        print(f"Processed {count} unique rows")
        print(f"Output saved to: {output_file if output_file else input_file.replace('.csv', '_cleaned.csv')}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to check for missing values.
                                  If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed.
    """
    if columns is None:
        return df.dropna()
    else:
        return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with their mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to fill missing values
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame containing only the outlier rows.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_column(df, column):
    """
    Standardize a column using z-score normalization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_standardized = df.copy()
    mean = df_standardized[column].mean()
    std = df_standardized[column].std()
    
    if std > 0:
        df_standardized[column] = (df_standardized[column] - mean) / std
    
    return df_standardized

def clean_dataframe(df, missing_strategy='remove', outlier_strategy='remove', columns_to_clean=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'remove', 'mean', 'ignore'
        outlier_strategy (str): Strategy for handling outliers.
                               Options: 'remove', 'detect', 'ignore'
        columns_to_clean (list, optional): Specific columns to clean.
                                          If None, cleans all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if columns_to_clean is None:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df, columns_to_clean)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df, columns_to_clean)
    
    if outlier_strategy == 'remove':
        for col in columns_to_clean:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate a summary of data quality metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing data quality metrics.
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'duplicate_rows': df.duplicated().sum()
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary[f'{col}_mean'] = df[col].mean()
        summary[f'{col}_std'] = df[col].std()
        summary[f'{col}_min'] = df[col].min()
        summary[f'{col}_max'] = df[col].max()
    
    return summaryimport pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        df = df.drop_duplicates()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].str.strip()
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    clean_csv(sys.argv[1], sys.argv[2])