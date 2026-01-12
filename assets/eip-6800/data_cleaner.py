import pandas as pd
import numpy as np

def normalize_column(series, method='minmax'):
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'zscore':
        return (series - series.mean()) / series.std()
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")

def remove_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns, normalization_method='minmax', outlier_multiplier=1.5):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
            cleaned_df[col] = normalize_column(cleaned_df[col], normalization_method)
    
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(2, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Normalized ranges - Feature A: [{cleaned['feature_a'].min():.3f}, {cleaned['feature_a'].max():.3f}]")
    print(f"Normalized ranges - Feature B: [{cleaned['feature_b'].min():.3f}, {cleaned['feature_b'].max():.3f}]")