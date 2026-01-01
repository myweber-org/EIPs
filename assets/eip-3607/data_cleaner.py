
import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    numeric_cols = ['feature1', 'feature2']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(result.head())
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def zscore_normalize(df, column):
    """Normalize column using z-score normalization."""
    df[column + '_normalized'] = stats.zscore(df[column])
    return df

def minmax_normalize(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_scaled'] = (df[column] - min_val) / (max_val - min_val)
    return df

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.dropna()

def clean_dataset(df, numeric_columns):
    """Main cleaning pipeline."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = zscore_normalize(cleaned_df, col)
            cleaned_df = minmax_normalize(cleaned_df, col)
    
    cleaned_df = handle_missing_values(cleaned_df)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'feature3': np.random.uniform(0, 1, 200)
    })
    
    sample_data.iloc[10:15, 0] = np.nan
    sample_data.iloc[20:25, 1] = np.nan
    
    numeric_cols = ['feature1', 'feature2', 'feature3']
    cleaned_data = clean_dataset(sample_data, numeric_cols)
    
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", cleaned_data.shape)
    print("\nCleaned data summary:")
    print(cleaned_data.describe())import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
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
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, None],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                     If None, adds '_cleaned' suffix to input filename.
        fill_strategy (str): Strategy for filling missing values. 
                            Options: 'mean', 'median', 'zero', 'drop'
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print(f"Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} missing")
        
        if fill_strategy == 'drop':
            df = df.dropna()
            print(f"Dropped rows with missing values")
        elif fill_strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    if fill_strategy == 'mean':
                        fill_value = df[col].mean()
                    else:
                        fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value:.2f}")
        elif fill_strategy == 'zero':
            df = df.fillna(0)
            print("Filled missing values with 0")
    
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_cleaned.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    print(f"Final data shape: {df.shape}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
        return validation_results
    
    validation_results['summary']['rows'] = len(df)
    validation_results['summary']['columns'] = len(df.columns)
    validation_results['summary']['missing_values'] = int(df.isnull().sum().sum())
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['summary']['numeric_columns'] = len(numeric_cols)
        
        for col in numeric_cols:
            if df[col].isnull().any():
                validation_results['issues'].append(f'Column {col} has missing values')
            
            if (df[col] < 0).any() and 'amount' in col.lower():
                validation_results['issues'].append(f'Column {col} contains negative values')
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 20.1],
        'category': ['A', 'B', 'A', None, 'C', 'C'],
        'score': [85, 92, None, 78, 88, 88]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    print("Testing data cleaning utility...")
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    print("\nValidating cleaned data...")
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
    print(f"Validation valid: {validation['is_valid']}")
    print(f"Validation summary: {validation['summary']}")
    
    if validation['issues']:
        print(f"Validation issues: {validation['issues']}")