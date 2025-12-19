import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(input_file='data/btcusdt.csv', output_file='data/btcusdt_clean.csv'):
    """Clean duplicate timestamps from data."""
    print(f"Cleaning {input_file}...")
    
    try:
        # Read the entire CSV file
        print(f"Reading CSV file...")
        df = pd.read_csv(input_file)
        
        # Display original stats
        print(f"\nOriginal data stats:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Convert datetime column
        print(f"\nConverting datetime column...")
        if 'datetime' in df.columns:
            # Parse with UTC timezone
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
            
            # Check for NaT values (failed conversions)
            nat_count = df['datetime'].isna().sum()
            if nat_count > 0:
                print(f"  Warning: {nat_count} rows have invalid datetime values")
                df = df.dropna(subset=['datetime'])
            
            df.set_index('datetime', inplace=True)
            print(f"  Set 'datetime' as index")
        else:
            print(f"  ERROR: No 'datetime' column found!")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
        
        # Display datetime range
        print(f"\nDatetime range:")
        print(f"  Start: {df.index.min()}")
        print(f"  End: {df.index.max()}")
        print(f"  Total days: {(df.index.max() - df.index.min()).days}")
        
        # Check for duplicates
        duplicate_count = df.index.duplicated().sum()
        print(f"\nChecking for duplicates:")
        print(f"  Duplicate timestamps: {duplicate_count:,}")
        print(f"  Percentage: {duplicate_count/len(df)*100:.2f}%")
        
        if duplicate_count > 0:
            # Show some duplicate examples
            duplicates = df[df.index.duplicated(keep=False)]
            print(f"\nDuplicate examples (first 10):")
            print(duplicates.head(10))
            
            # Remove duplicates (keep first occurrence)
            df = df[~df.index.duplicated(keep='first')]
            print(f"  Removed {duplicate_count:,} duplicates")
        
        # Sort by index (just in case)
        df = df.sort_index()
        
        # Standardize column names
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Verify column names
        print(f"\nFinal column names: {df.columns.tolist()}")
        
        # Check data quality
        print(f"\nData quality check:")
        print(f"  Missing values:")
        for col in df.columns:
            missing = df[col].isna().sum()
            print(f"    {col}: {missing:,} missing ({missing/len(df)*100:.2f}%)")
        
        # Remove timezone from index for compatibility
        print(f"\nProcessing index...")
        df.index = df.index.tz_localize(None)
        print(f"  Removed timezone from index")
        
        # Final stats
        print(f"\nCleaned data stats:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Start date: {df.index.min()}")
        print(f"  End date: {df.index.max()}")
        print(f"  Date range: {(df.index.max() - df.index.min()).days} days")
        
        # Check if we have recent data
        recent_date = pd.Timestamp('2024-01-01')
        recent_data = df[df.index >= recent_date]
        print(f"\nData since {recent_date}:")
        print(f"  Rows: {len(recent_data):,}")
        if len(recent_data) > 0:
            print(f"  Start: {recent_data.index.min()}")
            print(f"  End: {recent_data.index.max()}")
        else:
            print(f"  WARNING: No data found since {recent_date}!")
        
        # Save to CSV
        print(f"\nSaving cleaned data to {output_file}...")
        df.to_csv(output_file)
        print(f"  Saved {len(df):,} rows")
        
        # Create a small sample for verification
        sample_size = min(20, len(df))
        sample_start = df.head(sample_size)
        sample_end = df.tail(sample_size)
        
        print(f"\nSample data (first {sample_size} rows):")
        print(sample_start[['Open', 'High', 'Low', 'Close', 'Volume']].head())
        
        print(f"\nSample data (last {sample_size} rows):")
        print(sample_end[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
        
        print(f"\n{'='*60}")
        print("CLEANING COMPLETE!")
        print(f"{'='*60}")
        
        return df
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_data(filepath):
    """Verify the cleaned data file."""
    print(f"\n{'='*60}")
    print("VERIFYING CLEANED DATA")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} rows from {filepath}")
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        print(f"Index range: {df.index.min()} to {df.index.max()}")
        print(f"Total days: {(df.index.max() - df.index.min()).days}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        print(f"Duplicates in cleaned file: {duplicates}")
        
        # Check data types
        print(f"\nData types:")
        print(df.dtypes)
        
        # Check recent data
        recent_date = pd.Timestamp('2024-01-01')
        recent_data = df[df.index >= recent_date]
        print(f"\nData since {recent_date}: {len(recent_data):,} rows")
        
        if len(recent_data) > 0:
            print(f"Most recent dates:")
            for date in recent_data.index[-5:]:
                print(f"  {date}")
        
        return df
        
    except Exception as e:
        print(f"Verification failed: {e}")
        return None

if __name__ == '__main__':
    print("Starting data cleaning process...")
    print(f"{'='*60}")
    
    # Clean the data
    cleaned_df = clean_data('data/btcusdt.csv', 'data/btcusdt_clean.csv')
    
    if cleaned_df is not None:
        # Verify the cleaned data
        verify_data('data/btcusdt_clean.csv')
    else:
        print("\nData cleaning failed!")