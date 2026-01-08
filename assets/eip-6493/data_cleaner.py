
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

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = handle_missing_values(cleaned_df[[col]], strategy=missing_strategy)
            
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    raw_data = load_dataset(input_file)
    cleaned_data = clean_dataset(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, output_file)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif fill_missing == 'mode':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'x', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned)
    print(f"\nValidation: {message}")