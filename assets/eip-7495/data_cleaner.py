import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, drop_na=True, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to input CSV file.
    output_path (str, optional): Path for cleaned output CSV. If None, overwrites input.
    drop_na (bool): If True, drop rows with missing values. If False, fill them.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        print(f"Removed {duplicates_removed} duplicate rows.")
        
        # Handle missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            if drop_na:
                df.dropna(inplace=True)
                print("Dropped rows with missing values.")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if fill_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df[col].mode()[0]
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            fill_value = df[col].mean()
                        
                        df[col].fillna(fill_value, inplace=True)
                
                # For non-numeric columns, fill with mode
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    if df[col].isnull().any():
                        mode_value = df[col].mode()
                        if not mode_value.empty:
                            df[col].fillna(mode_value[0], inplace=True)
                        else:
                            df[col].fillna('Unknown', inplace=True)
                
                print(f"Filled missing values using '{fill_strategy}' strategy.")
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Save or return results
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        else:
            df.to_csv(file_path, index=False)
            print(f"Cleaned data saved to original file: {file_path}")
        
        print(f"Final data shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        infinite_count = np.isinf(numeric_df).sum().sum()
        if infinite_count > 0:
            print(f"Warning: Found {infinite_count} infinite values in numeric columns.")
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 40, 50, 50],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 
                                output_path='cleaned_sample.csv',
                                drop_na=False,
                                fill_strategy='mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
        print(f"Validation result: {validation_result}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    stats_dict = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            stats_dict[column] = stats
    
    return cleaned_df, stats_dict

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    # Introduce some outliers
    sample_data['value'][95] = 500
    sample_data['value'][96] = -200
    sample_data['score'][97] = 150
    sample_data['score'][98] = -50
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df[['value', 'score']].describe())
    
    cleaned_df, stats = clean_dataset(df, ['value', 'score'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")