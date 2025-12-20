
import pandas as pd
import numpy as np
from scipy import stats

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

def calculate_statistics(df):
    stats_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': stats.skew(df[col].dropna())
        }
    return pd.DataFrame(stats_dict).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'pressure': np.random.normal(1013, 10, 100)
    })
    
    numeric_cols = ['temperature', 'humidity', 'pressure']
    cleaned_data = clean_dataset(sample_data, numeric_cols)
    statistics = calculate_statistics(cleaned_data)
    
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", cleaned_data.shape)
    print("\nStatistics:")
    print(statistics)
import pandas as pd
import numpy as np

def remove_outliers(df, column, method='iqr', threshold=1.5):
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < threshold]
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def normalize_column(df, column, method='minmax'):
    if method == 'minmax':
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    elif method == 'standard':
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")
    return df

def clean_dataset(file_path, output_path, outlier_columns=None, normalize_columns=None):
    df = pd.read_csv(file_path)
    
    if outlier_columns:
        for col in outlier_columns:
            df = remove_outliers(df, col)
    
    if normalize_columns:
        for col in normalize_columns:
            df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset(
        'raw_data.csv',
        'cleaned_data.csv',
        outlier_columns=['age', 'income'],
        normalize_columns=['score', 'rating']
    )