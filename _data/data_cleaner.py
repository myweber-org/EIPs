import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning function
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    outlier_report = {}
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            filtered_data, outliers_removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            outlier_report[col] = outliers_removed
            cleaned_df = filtered_data
            
            # Normalize using z-score
            cleaned_df[f"{col}_normalized"] = z_score_normalize(cleaned_df, col)
    
    return cleaned_df, outlier_report

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    validation_results = {
        'has_required_columns': all(col in df.columns for col in required_columns),
        'row_count': len(df),
        'has_min_rows': len(df) >= min_rows,
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return validation_results

def create_sample_dataset(size=1000):
    """
    Create sample dataset for testing
    """
    np.random.seed(42)
    
    data = {
        'feature_a': np.random.normal(100, 15, size),
        'feature_b': np.random.exponential(2.0, size),
        'feature_c': np.random.uniform(0, 1, size),
        'category': np.random.choice(['A', 'B', 'C'], size)
    }
    
    # Add some outliers
    outlier_indices = np.random.choice(size, size=10, replace=False)
    data['feature_a'][outlier_indices] *= 5
    
    return pd.DataFrame(data)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df.reset_index(drop=True)

def main():
    data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 15, 18, 20, 22, 25, 28, 30,
                  100, 105, 110, 115, 120, 5, 8, 35, 40, 45]
    }
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['value'])
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")

if __name__ == "__main__":
    main()