
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    method (str): Method for outlier detection ('iqr' or 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df[column].dropna())
        abs_z_scores = abs(z_scores)
        filtered_df = df[abs_z_scores < threshold]
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return filtered_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
        'count': len(df[column]),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def process_numerical_columns(df, columns=None):
    """
    Process multiple numerical columns for outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numerical columns.
    
    Returns:
    pd.DataFrame: Processed DataFrame with outliers removed from specified columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    processed_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                processed_df = remove_outliers_iqr(processed_df, column)
            except Exception as e:
                print(f"Error processing column {column}: {e}")
    
    return processed_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'A'] = 500  # Add an outlier
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = process_numerical_columns(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to process, None for all numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to process, None for all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize, None for all numeric columns
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean = df_standardized[col].mean()
        std = df_standardized[col].std()
        if std > 0:
            df_standardized[col] = (df_standardized[col] - mean) / std
    
    return df_standardized

def clean_dataset(df, missing_strategy='mean', outlier_removal=True, standardization=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        outlier_removal (bool): Whether to remove outliers
        standardization (bool): Whether to standardize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = clean_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_removal:
        cleaned_df = remove_outliers_iqr(cleaned_df)
    
    if standardization:
        cleaned_df = standardize_columns(cleaned_df)
    
    return cleaned_df
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import re

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dictionary mapping old column names to new ones
        drop_duplicates: whether to remove duplicate rows
        normalize_text: whether to normalize text columns (lowercase, strip whitespace)
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
            df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df: pandas DataFrame
        email_column: name of the column containing email addresses
    
    Returns:
        DataFrame with additional 'email_valid' boolean column
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df_validated = df.copy()
    df_validated['email_valid'] = df_validated[email_column].str.match(email_pattern)
    
    valid_count = df_validated['email_valid'].sum()
    total_count = len(df_validated)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return df_validated

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: pandas DataFrame to save
        output_path: path to save the file
        format: output format ('csv', 'excel', or 'json')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path} in {format} format")
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding the threshold percentage.
    """
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold].reset_index(drop=True)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median for specified columns or all numeric columns.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col].fillna(median_val, inplace=True)
    
    return df_filled

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specified column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    """
    from sklearn.preprocessing import StandardScaler
    
    df_scaled = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled

def clean_dataset(df, missing_threshold=0.5, outlier_columns=None, outlier_multiplier=1.5):
    """
    Perform a complete cleaning pipeline on the dataset.
    """
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    df_clean = fill_missing_with_median(df_clean)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col, multiplier=outlier_multiplier)
    
    return df_clean
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100) * 2 + 5,
        'C': np.random.randn(100) * 0.5 - 2
    })
    sample_data.loc[0, 'A'] = 100
    sample_data.loc[1, 'B'] = -50
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")