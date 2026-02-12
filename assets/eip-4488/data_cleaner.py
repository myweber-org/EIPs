
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dictionary mapping old column names to new ones
        drop_duplicates: whether to remove duplicate rows
        normalize_text: whether to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email: string email address to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def extract_numeric(text):
    """
    Extract numeric values from text.
    
    Args:
        text: string containing numeric values
    
    Returns:
        List of numeric values found in text
    """
    numbers = re.findall(r'\d+\.?\d*', str(text))
    return [float(num) if '.' in num else int(num) for num in numbers]

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: pandas DataFrame to save
        output_path: path to save the file
        format: file format ('csv', 'excel', 'json')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path} in {format} format")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame and outlier indices.
    """
    cleaned_df = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col in df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = cleaned_df[(cleaned_df[col] < lower_bound) | 
                                 (cleaned_df[col] > upper_bound)].index
            outlier_indices.extend(outliers)
            
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & 
                                   (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df, list(set(outlier_indices))

def normalize_minmax(df, columns):
    """
    Normalize specified columns using min-max scaling.
    Returns DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    return normalized_df

def standardize_zscore(df, columns):
    """
    Standardize specified columns using z-score normalization.
    Returns DataFrame with standardized columns.
    """
    standardized_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = standardized_df[col].mean()
            std_val = standardized_df[col].std()
            
            if std_val > 0:
                standardized_df[col] = (standardized_df[col] - mean_val) / std_val
    
    return standardized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = df.columns
    
    handled_df = df.copy()
    
    for col in columns:
        if col in df.columns and handled_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = handled_df[col].mean()
            elif strategy == 'median':
                fill_value = handled_df[col].median()
            elif strategy == 'mode':
                fill_value = handled_df[col].mode()[0] if not handled_df[col].mode().empty else 0
            elif strategy == 'drop':
                handled_df = handled_df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            handled_df[col] = handled_df[col].fillna(fill_value)
    
    return handled_df

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    Returns validation results dictionary.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {},
        'shape': df.shape
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric.append(col)
        
        if non_numeric:
            validation_results['non_numeric_columns'] = non_numeric
            validation_results['is_valid'] = False
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    return validation_results
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a pandas DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df_clean = df.copy()
    df_clean[column_name] = df_clean[column_name].astype(str)
    df_clean[column_name] = df_clean[column_name].str.strip()
    df_clean[column_name] = df_clean[column_name].str.lower()
    df_clean = df_clean.drop_duplicates(subset=[column_name], keep='first')
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def remove_special_characters(text):
    """
    Remove special characters from a string, keeping only alphanumeric and spaces.
    """
    if not isinstance(text, str):
        return text
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def normalize_column(df, column_name):
    """
    Apply special character removal to a DataFrame column.
    """
    df[column_name] = df[column_name].apply(remove_special_characters)
    return df

if __name__ == "__main__":
    sample_data = {'Name': ['  Alice  ', 'Bob', 'alice', 'Charlie!', '  david  ']}
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, 'Name')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    normalized_df = normalize_column(cleaned_df, 'Name')
    print("\nNormalized DataFrame:")
    print(normalized_df)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        original_len = len(df)
        df = remove_outliers_iqr(df, col)
        removed_count = original_len - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', 'cleaned_data.csv')