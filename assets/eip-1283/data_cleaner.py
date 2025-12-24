import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
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

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_data = data.copy()
    outlier_report = {}
    normalization_report = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data, outliers_removed = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data, outliers_removed = remove_outliers_zscore(cleaned_data, col)
        else:
            raise ValueError("Invalid outlier method. Choose 'iqr' or 'zscore'")
        
        outlier_report[col] = outliers_removed
        
        if normalize_method == 'minmax':
            cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[f'{col}_normalized'] = normalize_zscore(cleaned_data, col)
        else:
            raise ValueError("Invalid normalize method. Choose 'minmax' or 'zscore'")
        
        normalization_report[col] = normalize_method
    
    return cleaned_data, outlier_report, normalization_report

def generate_cleaning_summary(data, cleaned_data, outlier_report, numeric_columns):
    """
    Generate a summary of the cleaning process.
    """
    summary = {
        'original_rows': len(data),
        'cleaned_rows': len(cleaned_data),
        'rows_removed': len(data) - len(cleaned_data),
        'outliers_removed': outlier_report,
        'numeric_columns_processed': numeric_columns,
        'cleaning_efficiency': (len(cleaned_data) / len(data)) * 100 if len(data) > 0 else 0
    }
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature1', 'feature2', 'feature3']
    
    cleaned_df, outliers, normalization = clean_dataset(
        sample_data, 
        numeric_cols, 
        outlier_method='iqr', 
        normalize_method='zscore'
    )
    
    summary = generate_cleaning_summary(sample_data, cleaned_df, outliers, numeric_cols)
    
    print(f"Original dataset shape: {sample_data.shape}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Outliers removed: {outliers}")
    print(f"Normalization applied: {normalization}")
    print(f"Cleaning efficiency: {summary['cleaning_efficiency']:.2f}%")
import pandas as pd

def clean_dataframe(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default is True.
        column_case (str): Target case for column names ('lower', 'upper', or 'title'). Default is 'lower'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle null values
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    else:
        # Fill numeric columns with median, object columns with mode
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove leading/trailing whitespace from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Replace spaces with underscores in column names
    cleaned_df.columns = cleaned_df.columns.str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        unique_columns (list): List of column names that should have unique values.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'duplicate_rows': df.duplicated().sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'unique_violations': {}
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    # Check unique constraints
    if unique_columns:
        for col in unique_columns:
            if col in df.columns:
                duplicates = df[col].duplicated().sum()
                if duplicates > 0:
                    validation_results['unique_violations'][col] = duplicates
                    validation_results['is_valid'] = False
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'Name': ['Alice', 'Bob', None, 'David'],
#         'Age': [25, None, 35, 40],
#         'City': ['NYC', 'LA', 'Chicago', None]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataframe(df, drop_na=False)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     validation = validate_dataframe(cleaned, required_columns=['name', 'age'])
#     print("\nValidation Results:")
#     print(validation)