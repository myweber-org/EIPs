
import pandas as pd

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping original column names to new names.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip whitespace, lowercase).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    # Normalize text columns
    if normalize_text:
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        print(f"Normalized {len(text_columns)} text columns.")
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns=None, allow_na_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of columns that must be present.
        allow_na_columns (list): List of columns where NA values are allowed.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'na_counts': {},
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['missing_columns'] = missing
            validation_result['is_valid'] = False
    
    # Check NA values
    na_counts = df.isna().sum()
    validation_result['na_counts'] = na_counts.to_dict()
    
    # Check if NA values are in disallowed columns
    if allow_na_columns is not None:
        disallowed_na = [col for col in df.columns 
                        if col not in allow_na_columns and na_counts[col] > 0]
        if disallowed_na:
            validation_result['is_valid'] = False
    
    return validation_result

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson ', 'ALICE'],
        'Age': [25, 30, 25, 35, None],
        'Email': ['john@example.com', 'jane@example.com', 
                 'john@example.com', 'bob@example.com', 'alice@example.com'],
        'City': ['New York', 'Los Angeles', 'New York', ' Chicago ', 'BOSTON']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, drop_duplicates=True, normalize_text=True)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_data(
        cleaned_df, 
        required_columns=['Name', 'Email'],
        allow_na_columns=['Age']
    )
    print("\nValidation Results:")
    print(validation)
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def process_csv(input_path, output_path, numeric_columns=None):
    df = pd.read_csv(input_path)
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    cleaned_df = clean_dataset(df, numeric_columns)
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df