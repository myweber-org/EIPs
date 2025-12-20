import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: multiplier for IQR (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with z-score normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    z_scores = (data[column] - mean_val) / std_val
    return z_scores

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if data.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def create_sample_data():
    """
    Create sample data for testing.
    
    Returns:
        pandas DataFrame with sample data
    """
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.uniform(0, 1, 100)
    }
    
    # Add some outliers
    data['value'][0] = 300
    data['value'][1] = -50
    
    # Add some missing values
    data['value'][5] = np.nan
    data['score'][10] = np.nan
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, 'value')
    print("After outlier removal:", cleaned_data.shape)
    
    # Normalize
    sample_data['value_normalized'] = normalize_minmax(sample_data, 'value')
    sample_data['value_zscore'] = z_score_normalize(sample_data, 'value')
    
    # Handle missing values
    filled_data = handle_missing_values(sample_data, strategy='mean')
    print("Missing values handled")
    
    # Validate
    is_valid, message = validate_dataframe(filled_data, ['id', 'value'])
    print(f"Data validation: {is_valid}, Message: {message}")import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): List of column names to consider for identifying duplicates
    """
    try:
        df = pd.read_csv(input_file)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_clean = df.drop_duplicates(keep='first')
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        print(f"Removed {len(df) - len(df_clean)} duplicate rows")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file] [--subset col1,col2,...]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    subset = None
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--subset" and i + 1 < len(sys.argv):
            subset = sys.argv[i + 1].split(',')
            break
        elif not sys.argv[i].startswith("--"):
            output_file = sys.argv[i]
    
    remove_duplicates(input_file, output_file, subset)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [3, 1, 2, 1, 4, 3, 5, 2, 6]
    cleaned = remove_duplicates_preserve_order(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
import pandas as pd
import re
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows from the dataframe."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def normalize_text(self, column: str) -> pd.DataFrame:
        """Normalize text in specified column: lowercase and remove extra spaces."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        self.df[column] = self.df[column].astype(str).apply(
            lambda x: re.sub(r'\s+', ' ', x.strip().lower())
        )
        print(f"Normalized text in column '{column}'")
        return self.df
    
    def fill_missing_values(self, column: str, value: str = 'unknown') -> pd.DataFrame:
        """Fill missing values in specified column with given value."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        missing_count = self.df[column].isnull().sum()
        self.df[column] = self.df[column].fillna(value)
        print(f"Filled {missing_count} missing values in column '{column}' with '{value}'")
        return self.df
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned dataframe."""
        return self.df

def clean_dataset(input_file: str, output_file: str) -> None:
    """Main function to clean a dataset from input file and save to output file."""
    try:
        df = pd.read_csv(input_file)
        cleaner = DataCleaner(df)
        
        cleaner.remove_duplicates()
        cleaner.normalize_text('description')
        cleaner.fill_missing_values('category')
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_data(df):
    """
    Validate that the cleaned DataFrame has no missing values.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("Data validation passed: No missing values found.")
        return True
    else:
        print(f"Data validation failed: {missing_values} missing values found.")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation_result = validate_data(cleaned_df)
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
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def main():
    # Example usage
    data = {'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11]}
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_summary_statistics(cleaned_df, 'values'))

if __name__ == "__main__":
    main()