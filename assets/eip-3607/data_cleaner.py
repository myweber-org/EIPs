import pandas as pd

def clean_dataset(df, column_name):
    """
    Clean a specific column in a pandas DataFrame.
    Removes duplicates, strips whitespace, and converts to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Remove duplicate rows based on the specified column
    df_cleaned = df.drop_duplicates(subset=[column_name])

    # Normalize the string values in the specified column
    df_cleaned[column_name] = df_cleaned[column_name].astype(str).str.strip().str.lower()

    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column using a simple regex pattern.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")

    df['is_valid_email'] = df[email_column].astype(str).apply(lambda x: bool(re.match(pattern, x)))
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'bob  '],
        'email': ['alice@example.com', 'bob@test.org', 'alice@example.com', 'invalid-email', 'BOB@TEST.ORG']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataset(df, 'name')
    print("\nDataFrame after cleaning 'name' column:")
    print(cleaned_df)

    validated_df = validate_email_column(cleaned_df, 'email')
    print("\nDataFrame with email validation:")
    print(validated_df[['name', 'email', 'is_valid_email']])import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
        outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers
    if outlier_method == 'iqr':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[column].quantile(0.25)
            Q3 = cleaned_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df[column] = cleaned_df[column].clip(lower_bound, upper_bound)
    
    elif outlier_method == 'zscore':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
            cleaned_df = cleaned_df[z_scores < 3]
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        validation_results['warnings'].append('No numeric columns found in DataFrame')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print("Validation Results:")
    print(validation)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result