
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if data.empty:
        return {}
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return stats

def process_dataset(df, numeric_columns):
    """
    Process dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return cleaned_df, all_statsimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
        factor: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling to [0, 1] range.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def clean_dataset(df, outlier_columns=None, normalize_columns=None, outlier_factor=1.5):
    """
    Complete data cleaning pipeline with outlier removal and normalization.
    
    Args:
        df: pandas DataFrame
        outlier_columns: columns for outlier removal
        normalize_columns: columns for normalization
        outlier_factor: IQR factor for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    print(f"Original dataset shape: {df.shape}")
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df, outlier_columns, outlier_factor)
    print(f"After outlier removal: {df_clean.shape}")
    
    # Normalize data
    df_final = normalize_minmax(df_clean, normalize_columns)
    
    return df_final

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    # Add some outliers
    data['feature1'][::100] = np.random.uniform(500, 1000, 10)
    
    df = pd.DataFrame(data)
    
    # Clean the dataset
    cleaned_df = clean_dataset(
        df,
        outlier_columns=['feature1', 'feature2', 'feature3'],
        normalize_columns=['feature1', 'feature2', 'feature3']
    )
    
    print(f"Final cleaned shape: {cleaned_df.shape}")
    print("\nSummary statistics:")
    print(cleaned_df.describe())
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove rows where all values are NaN
        df = df.dropna(how='all')
        
        print(f"Final cleaned data shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate the cleaned dataframe for basic integrity.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for any remaining NaN values
    if df.isnull().sum().sum() > 0:
        print(f"Validation warning: DataFrame still contains {df.isnull().sum().sum()} NaN values.")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Validation warning: Column '{col}' contains infinite values.")
    
    print("Data validation completed successfully.")
    return True

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        # Validate with example required columns
        required_cols = ['id', 'value', 'category']
        validate_data(cleaned_df, required_cols)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled. Default is None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, unique_columns=None):
    """
    Validate the DataFrame for required columns and unique constraints.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    unique_columns (list): List of column names that should have unique values.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'duplicate_values': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns:
                duplicates = df[df.duplicated(subset=[col], keep=False)]
                if not duplicates.empty:
                    validation_results['duplicate_values'][col] = duplicates[col].unique().tolist()
                    validation_results['is_valid'] = False
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'age': [25, 30, 30, None, 35],
        'score': [85.5, 92.0, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    validation = validate_data(cleaned, required_columns=['id', 'name'], unique_columns=['id'])
    print("Validation Results:")
    print(validation)