import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean', date_columns=None):
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        date_columns: List of column names to parse as dates
    """
    
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Parse date columns if specified
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'mode':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        
        # Fill non-numeric columns with empty string
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna('')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed. Cleaned data saved to: {output_file}")
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Args:
        df: Pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', None, '2023-01-05'],
        'temperature': [22.5, 23.1, None, 24.3, 25.0],
        'humidity': [65, 68, 70, None, 72],
        'location': ['A', 'B', 'A', 'C', 'B']
    }
    
    # Create and save sample CSV
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data(
        input_file='sample_data.csv',
        output_file='cleaned_data.csv',
        missing_strategy='mean',
        date_columns=['date']
    )
    
    # Validate cleaned data
    if cleaned_df is not None:
        is_valid = validate_dataframe(
            cleaned_df,
            required_columns=['date', 'temperature', 'humidity'],
            min_rows=3
        )
        print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")
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
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def main():
    data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    data.loc[np.random.choice(1000, 50), 'feature_a'] = np.random.uniform(500, 1000, 50)
    
    cleaned = clean_dataset('sample_data.csv', ['feature_a', 'feature_b'])
    cleaned.to_csv('cleaned_data.csv', index=False)
    print(f"Original shape: {data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")

if __name__ == "__main__":
    main()