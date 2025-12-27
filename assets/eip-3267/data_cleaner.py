
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values, converting data types,
    and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        # Convert date columns if present
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    continue
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        # Generate cleaning report
        report = {
            'input_file': input_file,
            'output_file': output_file,
            'timestamp': datetime.now().isoformat(),
            'initial_rows': initial_count,
            'final_rows': len(df),
            'duplicates_removed': duplicates_removed,
            'missing_values_filled': df.isnull().sum().sum(),
            'columns_processed': list(df.columns)
        }
        
        return report
        
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        return None

def validate_data(df):
    """
    Validate data quality after cleaning.
    """
    validation_results = {}
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    validation_results['remaining_nulls'] = null_counts.to_dict()
    
    # Check data types
    validation_results['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Check basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['statistics'] = df[numeric_cols].describe().to_dict()
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaning_report = clean_csv_data(input_csv, output_csv)
    
    if cleaning_report:
        print("Data cleaning completed successfully!")
        print(f"Report: {cleaning_report}")
        
        # Load and validate cleaned data
        cleaned_df = pd.read_csv(output_csv)
        validation = validate_data(cleaned_df)
        print(f"Validation results: {validation}")
    else:
        print("Data cleaning failed!")