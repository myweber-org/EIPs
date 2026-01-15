
import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean the input DataFrame by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values: fill numeric columns with median, categorical with mode
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in [np.float64, np.int64]:
            df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
        else:
            df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown', inplace=True)
    
    return df_cleaned

def validate_data(df):
    """
    Validate the cleaned DataFrame for any remaining issues.
    """
    validation_report = {}
    validation_report['total_rows'] = len(df)
    validation_report['total_columns'] = len(df.columns)
    validation_report['missing_values'] = df.isnull().sum().sum()
    validation_report['duplicate_rows'] = df.duplicated().sum()
    
    return validation_report

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, 2, 4, np.nan],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', np.nan]
    })
    
    cleaned_df = clean_data(sample_data)
    report = validate_data(cleaned_df)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\nValidation Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by converting to float and removing NaN values.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to clean
    
    Returns:
        DataFrame with cleaned numeric column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    original_count = len(df)
    df = df.dropna(subset=[column_name])
    
    removed_count = original_count - len(df)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with invalid values in '{column_name}'")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
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

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': ['95', '88', '88', '72', 'invalid', '65', '91']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df = remove_duplicates(df, subset=['id', 'name'], keep='first')
    df = clean_numeric_column(df, 'score')
    
    print("Cleaned DataFrame:")
    print(df)
    print()
    
    is_valid, message = validate_dataframe(df, required_columns=['id', 'name', 'score'])
    print(f"Validation: {is_valid} - {message}")

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Cleans a pandas DataFrame by removing duplicates,
    standardizing column names, and filling missing values.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Standardize column names: lower case, replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')

    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()

    # Fill missing numeric values with column median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fill missing categorical values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown')

    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

def validate_data(df, required_columns):
    """
    Validates that the DataFrame contains all required columns.
    Returns True if valid, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4],
        'Order Value': [100.0, 200.0, 200.0, np.nan, 400.0],
        'Product Category': ['A', 'B', 'B', None, 'C'],
        'Region': ['North', 'South', 'South', 'East', 'West']
    }

    df_raw = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_raw)
    print("\n")

    df_cleaned = clean_dataset(df_raw)
    print("Cleaned DataFrame:")
    print(df_cleaned)
    print("\n")

    required_cols = ['customer_id', 'order_value', 'product_category']
    is_valid = validate_data(df_cleaned, required_cols)
    print(f"Data validation result: {is_valid}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
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
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
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
    print("Original summary statistics:")
    original_stats = calculate_summary_statistics(df, 'value')
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    removed_count = len(df) - len(cleaned_df)
    print(f"\nRemoved {removed_count} outliers")

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
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
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_dfimport csv
import sys

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Remove duplicate rows from a CSV file based on specified key columns.
    If key_columns is None, all columns are used for duplicate detection.
    """
    seen = set()
    unique_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            
            if key_columns is None:
                key_columns = list(range(len(headers)))
            else:
                key_columns = [headers.index(col) for col in key_columns]
            
            for row in reader:
                key = tuple(row[i] for i in key_columns)
                if key not in seen:
                    seen.add(key)
                    unique_rows.append(row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(unique_rows)
        
        print(f"Processed {len(unique_rows)} unique rows from {input_file}")
        print(f"Output saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_file> <output_file> [key_column1 key_column2 ...]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_columns = sys.argv[3:] if len(sys.argv) > 3 else None
    
    remove_duplicates(input_file, output_file, key_columns)import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    """Main function to clean dataset."""
    df = load_data(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_cols:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')