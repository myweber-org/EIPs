import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def impute_missing_with_median(data, column):
    median_value = data[column].median()
    data[column].fillna(median_value, inplace=True)
    return data

def remove_duplicates(data):
    return data.drop_duplicates()

def standardize_column(data, column):
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std
    return data

def clean_dataset(data, numeric_columns):
    cleaned_data = data.copy()
    cleaned_data = remove_duplicates(cleaned_data)
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = impute_missing_with_median(cleaned_data, col)
            cleaned_data = standardize_column(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 50, 200],
        'C': ['x', 'y', 'z', 'x', 'y', 'y', 'z']
    })
    
    print("Original Data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data, ['A', 'B'])
    print("\nCleaned Data:")
    print(cleaned)
    
    outliers_a = detect_outliers_iqr(sample_data, 'A')
    print("\nOutliers in column A:")
    print(outliers_a)