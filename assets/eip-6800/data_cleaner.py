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
import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path):
    """
    Load CSV data, handle missing values, and convert data types.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df):
    """
    Perform basic data validation checks.
    """
    if df is None or df.empty:
        return False
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"{key}: {value}")
    
    return validation_results['missing_values'] == 0

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_data(cleaned_df)
        if is_valid:
            print("Data validation passed.")
        else:
            print("Data validation failed - check for issues.")