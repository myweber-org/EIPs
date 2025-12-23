
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    cleaned_df = df.copy()
    summary_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            summary_stats[column] = calculate_summary_statistics(cleaned_df, column)
            summary_stats[column]['outliers_removed'] = removed_count
    
    return cleaned_df, summary_stats

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    columns_to_clean = ['temperature', 'humidity', 'pressure']
    cleaned_df, stats = clean_dataset(df, columns_to_clean)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    print("Summary Statistics:")
    for column, column_stats in stats.items():
        print(f"\n{column}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value}")import pandas as pd
import re

def clean_dataframe(df, text_columns=None, drop_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        text_columns (list): List of column names containing text to standardize.
                             If None, all object dtype columns are processed.
        drop_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if text_columns is None:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(_standardize_text)
    
    return cleaned_df

def _standardize_text(text):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text: Input text (string or other type).
    
    Returns:
        str: Standardized text or original value if not a string.
    """
    if not isinstance(text, str):
        return text
    
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with valid emails and a validation flag.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_validated = df.copy()
    df_validated['email_valid'] = df_validated[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = df_validated['email_valid'].sum()
    print(f"Found {valid_count} valid email addresses out of {len(df_validated)} rows.")
    
    return df_validated

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.com'],
        'notes': ['  Some  notes  ', 'More notes!', 'Duplicate entry', '  Some  notes  ']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df, text_columns=['name', 'notes'])
    print(cleaned)
    print("\nValidated Emails:")
    validated = validate_email_column(cleaned, 'email')
    print(validated[['email', 'email_valid']])import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization_method='zscore'):
    """
    Clean dataset by removing outliers and applying normalization.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'zscore':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content.
    """
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    try:
        validate_data(df, numeric_cols)
        cleaned_df = clean_dataset(df, numeric_cols, normalization_method='zscore')
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print(f"Cleaned dataset summary:\n{cleaned_df.describe()}")
    except ValueError as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    main()
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result