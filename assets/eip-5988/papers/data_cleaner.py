
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    """
    clean_data = data.copy()
    for col in columns:
        outliers = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        if max_val > min_val:
            normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    """
    standardized_data = data.copy()
    for col in columns:
        mean_val = standardized_data[col].mean()
        std_val = standardized_data[col].std()
        if std_val > 0:
            standardized_data[col] = (standardized_data[col] - mean_val) / std_val
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    if columns is None:
        columns = data.columns
    
    filled_data = data.copy()
    
    for col in columns:
        if filled_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = filled_data[col].mean()
            elif strategy == 'median':
                fill_value = filled_data[col].median()
            elif strategy == 'mode':
                fill_value = filled_data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            filled_data[col] = filled_data[col].fillna(fill_value)
    
    return filled_data

def clean_dataset(data, numerical_cols, outlier_threshold=1.5, 
                  normalize=True, standardize=False, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy, columns=numerical_cols)
    
    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, numerical_cols, threshold=outlier_threshold)
    
    # Apply normalization if requested
    if normalize:
        cleaned_data = normalize_minmax(cleaned_data, numerical_cols)
    
    # Apply standardization if requested (overrides normalization for those columns)
    if standardize:
        cleaned_data = standardize_zscore(cleaned_data, numerical_cols)
    
    return cleaned_data

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some missing values
    sample_data.loc[10:15, 'feature1'] = np.nan
    sample_data.loc[20:25, 'feature2'] = np.nan
    
    # Clean the dataset
    numerical_columns = ['feature1', 'feature2', 'feature3']
    cleaned = clean_dataset(
        sample_data, 
        numerical_cols=numerical_columns,
        outlier_threshold=1.5,
        normalize=True,
        standardize=False,
        missing_strategy='mean'
    )
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Missing values after cleaning: {cleaned[numerical_columns].isnull().sum().sum()}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns):
    """Normalize data using min-max scaling."""
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            elif strategy == 'median':
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            elif strategy == 'mode':
                df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        else:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    return df_filled

def clean_dataset(input_path, output_path, numeric_columns):
    """Complete data cleaning pipeline."""
    df = load_dataset(input_path)
    df = handle_missing_values(df, strategy='median')
    df = remove_outliers_iqr(df, numeric_columns)
    df = normalize_data(df, numeric_columns)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    numeric_cols = ['age', 'income', 'score']
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv', numeric_cols)
    print(f"Data cleaning completed. Shape: {cleaned_df.shape}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw_data = load_dataset('raw_data.csv')
    numeric_cols = ['age', 'income', 'score']
    cleaned_data = clean_data(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, 'cleaned_data.csv')
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply string normalization.
                                          If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    
    # Identify columns for string cleaning
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    print(f"Removed {removed_duplicates} duplicate rows.")
    print(f"Cleaned columns: {columns_to_clean}")
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters from the edges.
    
    Args:
        text (str): Input string to normalize.
    
    Returns:
        str: Normalized string, or original value if not a string.
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using a simple regex pattern.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df_valid = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_valid['email_valid'] = df_valid[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = df_valid['email_valid'].sum()
    total_count = len(df_valid)
    
    print(f"Email validation: {valid_count} valid out of {total_count} total.")
    
    return df_valid

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice Johnson  ', 'BOB BROWN'],
        'email': ['john@example.com', 'jane.smith@test.org', 'invalid-email', 'alice@company.co.uk', 'bob@domain.com'],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_clean = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(df_clean)
    print("\n")
    
    # Validate emails
    df_validated = validate_email_column(df_clean, 'email')
    print("\nDataFrame with email validation:")
    print(df_validated[['name', 'email', 'email_valid']])
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result