
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method for specified columns."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method for specified columns."""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(df, columns):
    """Normalize specified columns using Min-Max scaling."""
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """Normalize specified columns using Z-score standardization."""
    df_norm = df.copy()
    for col in columns:
        mean_val = df_norm[col].mean()
        std_val = df_norm[col].std()
        df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def handle_missing_values(df, strategy='mean'):
    """Handle missing values using specified strategy."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif strategy == 'mode':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mode().iloc[0])
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    return df_clean

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """Main function to clean dataset with multiple preprocessing steps."""
    df_clean = df.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, numeric_columns)
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean, numeric_columns)
    
    return df_clean

def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV file."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        raw_data = load_dataset(input_file)
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
        
        cleaned_data = clean_dataset(
            raw_data, 
            numeric_columns=numeric_cols,
            outlier_method='iqr',
            normalize_method='minmax',
            missing_strategy='mean'
        )
        
        save_cleaned_data(cleaned_data, output_file)
        print(f"Data cleaning completed. Original shape: {raw_data.shape}, Cleaned shape: {cleaned_data.shape}")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")