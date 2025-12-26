import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def detect_skewed_columns(df, threshold=0.5):
    """
    Detect columns with significant skewness
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    
    for col in numeric_cols:
        skewness = stats.skew(df[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def log_transform_skewed(df, skewed_cols):
    """
    Apply log transformation to skewed columns
    """
    df_transformed = df.copy()
    
    for col, _ in skewed_cols:
        if df_transformed[col].min() > 0:
            df_transformed[col] = np.log1p(df_transformed[col])
        else:
            offset = abs(df_transformed[col].min()) + 1
            df_transformed[col] = np.log1p(df_transformed[col] + offset)
    
    return df_transformed

def clean_dataset(df, outlier_threshold=1.5, skew_threshold=0.5):
    """
    Complete data cleaning pipeline
    """
    print(f"Original shape: {df.shape}")
    
    df_clean = remove_outliers_iqr(df, threshold=outlier_threshold)
    print(f"After outlier removal: {df_clean.shape}")
    
    skewed = detect_skewed_columns(df_clean, threshold=skew_threshold)
    if skewed:
        print(f"Found {len(skewed)} skewed columns")
        df_clean = log_transform_skewed(df_clean, skewed)
    
    df_normalized = normalize_minmax(df_clean)
    
    return df_normalized

def validate_cleaning(df_original, df_cleaned):
    """
    Validate cleaning results
    """
    original_stats = df_original.describe()
    cleaned_stats = df_cleaned.describe()
    
    comparison = pd.DataFrame({
        'original_mean': original_stats.loc['mean'],
        'cleaned_mean': cleaned_stats.loc['mean'],
        'original_std': original_stats.loc['std'],
        'cleaned_std': cleaned_stats.loc['std']
    })
    
    return comparison