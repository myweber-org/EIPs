import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda x: isinstance(x, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda x: not x.empty, "DataFrame cannot be empty"),
        (lambda x: x.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check_func, error_msg in required_checks:
        if not check_func(df):
            raise ValueError(error_msg)
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2.0, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50)
    })
    
    print("Original shape:", sample_data.shape)
    validated = validate_dataframe(sample_data)
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned = clean_dataset(sample_data, numeric_cols)
    print("Cleaned shape:", cleaned.shape)
    print("Cleaned data summary:")
    print(cleaned[numeric_cols].describe())
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates, optionally filter by count threshold.
    If threshold is provided, only items appearing less than threshold times are kept.
    """
    if not data:
        return []
    
    if threshold is None:
        return remove_duplicates(data)
    
    from collections import Counter
    counter = Counter(data)
    result = [item for item in data if counter[item] < threshold]
    return remove_duplicates(result)

def validate_data(data):
    """
    Validate that data is a list and contains only hashable types.
    Raises TypeError if validation fails.
    """
    if not isinstance(data, list):
        raise TypeError("Input must be a list")
    
    for item in data:
        try:
            hash(item)
        except TypeError:
            raise TypeError(f"Item {item} is not hashable")
    
    return True

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    print("Threshold 2:", clean_data_with_threshold(sample_data, threshold=2))