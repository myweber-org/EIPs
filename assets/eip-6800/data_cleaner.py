
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Cleaned data saved to 'cleaned_data.csv'")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
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
    Normalize data using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization_method='minmax'):
    """
    Main cleaning function for numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Remove outliers
        cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
        
        # Apply normalization
        if normalization_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization_method == 'zscore':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with significant skewness.
    """
    skewed_cols = []
    for col in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def log_transform_skewed(data, skewed_columns):
    """
    Apply log transformation to reduce skewness.
    """
    transformed_data = data.copy()
    for col, _ in skewed_columns:
        if transformed_data[col].min() <= 0:
            # Add constant to handle zero/negative values
            transformed_data[col] = np.log(transformed_data[col] - transformed_data[col].min() + 1)
        else:
            transformed_data[col] = np.log(transformed_data[col])
    return transformed_data