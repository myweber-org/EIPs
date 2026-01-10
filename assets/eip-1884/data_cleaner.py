
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
    print(f"Data validation result: {is_valid}")