import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame
        drop_duplicates: Whether to drop duplicate rows
        fill_missing: Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets certain criteria.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def process_data(file_path, output_path=None):
    """
    Main function to process and clean data from a CSV file.
    
    Args:
        file_path: Path to input CSV file
        output_path: Path to save cleaned data (optional)
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        if not validate_data(df):
            return None
        
        cleaned_df = clean_dataset(df)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    result = process_data(input_file, output_file)
    
    if result is not None:
        print(f"Data cleaning completed. Shape: {result.shape}")
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
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    report = {
        'original_rows': len(data),
        'outliers_removed': 0,
        'columns_normalized': []
    }
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        if outlier_method == 'iqr':
            filtered_data, removed = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            filtered_data, removed = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        cleaned_data = filtered_data
        report['outliers_removed'] += removed
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
        
        report['columns_normalized'].append(column)
    
    report['final_rows'] = len(cleaned_data)
    report['removal_percentage'] = (report['outliers_removed'] / report['original_rows']) * 100
    
    return cleaned_data, report

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    validation_report = {
        'is_valid': True,
        'missing_columns': [],
        'null_values': {},
        'validation_errors': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            validation_report['missing_columns'] = missing
            validation_report['is_valid'] = False
            validation_report['validation_errors'].append(f"Missing required columns: {missing}")
    
    if not allow_nan:
        null_counts = data.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0].to_dict()
        if columns_with_nulls:
            validation_report['null_values'] = columns_with_nulls
            validation_report['is_valid'] = False
            validation_report['validation_errors'].append(f"Found null values in columns: {list(columns_with_nulls.keys())}")
    
    return validation_report