
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalization == 'zscore':
            cleaned_df = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 200, 100)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    cleaned_data = clean_dataset(
        sample_df, 
        numeric_columns=numeric_cols,
        outlier_removal=True,
        normalization='zscore'
    )
    
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Cleaned columns: {cleaned_data.columns.tolist()}")
    print(f"Sample statistics:\n{cleaned_data.describe()}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    return df.dropna()

def main():
    data = load_dataset('raw_data.csv')
    numeric_cols = ['age', 'income', 'score']
    cleaned_data = clean_data(data, numeric_cols)
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Original: {len(data)} rows, Cleaned: {len(cleaned_data)} rows")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return pd.DataFrame()

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep='first')
    removed = initial_count - len(df_clean)
    print(f"Removed {removed} duplicate records")
    return df_clean

def standardize_dates(df, date_columns):
    """Standardize date formats in specified columns."""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Standardized date format for column: {col}")
    return df

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled {missing_count} missing values in {col} using {strategy} strategy")
    
    return df

def clean_column_names(df):
    """Clean column names by removing whitespace and standardizing format."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print("Standardized column names")
    return df

def validate_data(df):
    """Perform basic data validation checks."""
    validation_results = {
        'total_records': len(df),
        'columns_with_missing': df.isnull().any().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'date_columns': len(df.select_dtypes(include=['datetime64']).columns),
        'duplicate_rows': df.duplicated().sum()
    }
    
    print("\nData Validation Results:")
    for key, value in validation_results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return validation_results

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def main():
    """Main function to execute data cleaning pipeline."""
    input_file = 'raw_data.csv'
    output_file = 'cleaned_data.csv'
    
    print("Starting data cleaning process...")
    
    # Load raw data
    df = load_data(input_file)
    
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # Data cleaning steps
    df = clean_column_names(df)
    df = remove_duplicates(df)
    df = standardize_dates(df, ['date', 'timestamp', 'created_at'])
    df = fill_missing_values(df, strategy='mean')
    
    # Validate cleaned data
    validation_results = validate_data(df)
    
    # Save cleaned data
    save_cleaned_data(df, output_file)
    
    print("\nData cleaning completed successfully!")

if __name__ == "__main__":
    main()