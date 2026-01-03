
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data distribution
    """
    skewness = data[column].skew()
    is_skewed = abs(skewness) > threshold
    
    return {
        'skewness': skewness,
        'is_skewed': is_skewed,
        'recommendation': 'log_transform' if is_skewed else 'no_transform'
    }

def log_transform(data, column, add_constant=1):
    """
    Apply log transformation to reduce skewness
    """
    transformed = np.log(data[column] + add_constant)
    return transformed

def clean_dataset(df, numeric_columns=None):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    cleaning_report = {}
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            outliers_removed = original_count - len(cleaned_df)
            
            # Check skewness
            skew_info = detect_skewness(cleaned_df, col)
            
            # Apply transformation if skewed
            if skew_info['is_skewed']:
                cleaned_df[f'{col}_log'] = log_transform(cleaned_df, col)
                transform_applied = True
            else:
                transform_applied = False
            
            # Normalize the column
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            
            cleaning_report[col] = {
                'outliers_removed': outliers_removed,
                'skewness': skew_info['skewness'],
                'transform_applied': transform_applied,
                'final_rows': len(cleaned_df)
            }
    
    return cleaned_df, cleaning_report

def validate_data_quality(df, required_columns=None):
    """
    Validate data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        quality_report['missing_required_columns'] = missing_columns
        quality_report['all_required_present'] = len(missing_columns) == 0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        quality_report['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return quality_report

# Example usage function
def process_data_file(filepath, output_path=None):
    """
    Complete data processing pipeline from file
    """
    try:
        # Read data
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        # Validate data quality
        quality_report = validate_data_quality(df)
        
        # Clean data
        cleaned_df, cleaning_report = clean_dataset(df)
        
        # Save processed data
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
        
        return {
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'quality_report': quality_report,
            'cleaning_report': cleaning_report,
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }import numpy as np
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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True