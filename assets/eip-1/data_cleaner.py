
import pandas as pd

def clean_dataset(df):
    """
    Remove duplicate rows and fill missing values with column mean.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned

def validate_data(df):
    """
    Validate that cleaned dataset has no duplicates and minimal missing values.
    """
    has_duplicates = df.duplicated().any()
    missing_count = df.isnull().sum().sum()
    
    return {
        'has_duplicates': has_duplicates,
        'missing_values': missing_count,
        'is_valid': not has_duplicates and missing_count == 0
    }

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.0, 20.0, None],
        'category': ['A', 'B', 'B', None, 'C']
    })
    
    cleaned_data = clean_dataset(sample_data)
    validation = validate_data(cleaned_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Validation results: {validation}")
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    df[column + '_normalized'] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_dataset(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')