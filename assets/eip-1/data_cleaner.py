import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing low-quality columns.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path for cleaned output CSV. If None, returns DataFrame.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
        drop_threshold (float): Drop columns with missing ratio above this threshold (0.0 to 1.0).
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else writes to file.
    """
    try:
        df = pd.read_csv(file_path)
        original_shape = df.shape
        print(f"Original data shape: {original_shape}")
        
        # Calculate missing ratios
        missing_ratios = df.isnull().sum() / len(df)
        
        # Drop columns with high missing ratio
        columns_to_drop = missing_ratios[missing_ratios > drop_threshold].index
        if len(columns_to_drop) > 0:
            print(f"Dropping columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
            df = df.drop(columns=columns_to_drop)
        
        # Fill remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if fill_strategy in ['mean', 'median'] and len(numeric_cols) > 0:
            if fill_strategy == 'mean':
                fill_values = df[numeric_cols].mean()
            else:
                fill_values = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)
        
        elif fill_strategy == 'mode' and len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        
        elif fill_strategy == 'zero' and len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill any remaining categorical NaNs with 'Unknown'
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows")
            df = df.drop_duplicates()
        
        final_shape = df.shape
        print(f"Cleaned data shape: {final_shape}")
        print(f"Removed {original_shape[0] - final_shape[0]} rows and {original_shape[1] - final_shape[1]} columns")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1],
        'category': ['A', 'B', np.nan, 'A', 'C'],
        'score': [85, 92, 78, np.nan, 88]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
        print(f"Data validation: {'Passed' if is_valid else 'Failed'}")
        
        import os
        os.remove('test_data.csv')