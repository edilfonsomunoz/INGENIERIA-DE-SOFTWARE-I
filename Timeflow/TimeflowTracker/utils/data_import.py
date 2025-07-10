import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine
import tempfile

def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = None

def load_csv_file(file):
    """Load CSV file with error handling"""
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV file: {str(e)}"

def load_excel_file(file):
    """Load Excel file with error handling"""
    try:
        df = pd.read_excel(file)
        return df, None
    except Exception as e:
        return None, f"Error loading Excel file: {str(e)}"

def load_json_file(file):
    """Load JSON file with error handling"""
    try:
        data = json.load(file)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return None, "Invalid JSON format"
        return df, None
    except Exception as e:
        return None, f"Error loading JSON file: {str(e)}"

def connect_to_database(db_type, connection_string):
    """Connect to database and return connection"""
    try:
        if db_type == "SQLite":
            conn = sqlite3.connect(connection_string)
            return conn, None
        elif db_type == "PostgreSQL":
            # Use environment variables for PostgreSQL connection
            db_url = os.getenv("DATABASE_URL", connection_string)
            engine = create_engine(db_url)
            return engine, None
        else:
            return None, f"Unsupported database type: {db_type}"
    except Exception as e:
        return None, f"Database connection error: {str(e)}"

def generate_simulated_data(num_observations, pattern_type="trend_seasonal"):
    """Generate simulated time series data"""
    try:
        # Create date range
        start_date = datetime.now() - timedelta(days=num_observations)
        dates = pd.date_range(start=start_date, periods=num_observations, freq='D')
        
        # Generate base series
        np.random.seed(42)  # For reproducibility
        t = np.arange(num_observations)
        
        if pattern_type == "trend_seasonal":
            # Trend component
            trend = 0.5 * t + 100
            
            # Seasonal component (weekly pattern)
            seasonal = 10 * np.sin(2 * np.pi * t / 7) + 5 * np.cos(2 * np.pi * t / 365.25)
            
            # Noise component
            noise = np.random.normal(0, 5, num_observations)
            
            # Combine components
            values = trend + seasonal + noise
            
        elif pattern_type == "stationary":
            # Stationary series with noise
            values = np.random.normal(100, 15, num_observations)
            
        elif pattern_type == "random_walk":
            # Random walk
            changes = np.random.normal(0, 1, num_observations)
            values = np.cumsum(changes) + 100
            
        else:
            # Default trend + seasonal
            trend = 0.5 * t + 100
            seasonal = 10 * np.sin(2 * np.pi * t / 7)
            noise = np.random.normal(0, 5, num_observations)
            values = trend + seasonal + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        return df, None
        
    except Exception as e:
        return None, f"Error generating simulated data: {str(e)}"

def validate_time_series_data(df):
    """Validate that the data is suitable for time series analysis"""
    issues = []
    
    # Check if DataFrame is empty
    if df.empty:
        issues.append("Dataset is empty")
        return issues
    
    # Check for date column
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
            date_columns.append(col)
    
    if not date_columns:
        # Try to identify potential date columns
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head())
                date_columns.append(col)
            except:
                continue
    
    if not date_columns:
        issues.append("No date/time column found")
    
    # Check for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        issues.append("No numeric columns found for analysis")
    
    # Check data size
    if len(df) < 20:
        issues.append("Dataset too small for reliable time series analysis (minimum 20 observations recommended)")
    
    return issues

def prepare_time_series_data(df, date_column, value_column):
    """Prepare data for time series analysis"""
    try:
        # Make a copy
        ts_df = df.copy()
        
        # Convert date column to datetime
        ts_df[date_column] = pd.to_datetime(ts_df[date_column])
        
        # Set date as index
        ts_df.set_index(date_column, inplace=True)
        
        # Sort by date
        ts_df.sort_index(inplace=True)
        
        # Select only the value column for analysis
        if value_column in ts_df.columns:
            ts_df = ts_df[[value_column]]
        
        return ts_df, None
        
    except Exception as e:
        return None, f"Error preparing time series data: {str(e)}"

def detect_data_frequency(df):
    """Detect the frequency of the time series data"""
    try:
        if len(df) < 2:
            return "Unknown"
        
        # Calculate differences between consecutive dates
        time_diffs = df.index.to_series().diff().dropna()
        
        # Find the most common difference
        most_common_diff = time_diffs.mode().iloc[0]
        
        # Map to frequency strings
        if most_common_diff == pd.Timedelta(days=1):
            return "Daily"
        elif most_common_diff == pd.Timedelta(days=7):
            return "Weekly"
        elif most_common_diff == pd.Timedelta(days=30) or most_common_diff == pd.Timedelta(days=31):
            return "Monthly"
        elif most_common_diff == pd.Timedelta(days=365) or most_common_diff == pd.Timedelta(days=366):
            return "Yearly"
        elif most_common_diff == pd.Timedelta(hours=1):
            return "Hourly"
        else:
            return f"Custom ({most_common_diff})"
            
    except Exception as e:
        return "Unknown"
