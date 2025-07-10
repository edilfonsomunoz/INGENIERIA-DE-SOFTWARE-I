import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def detect_outliers(data, method='iqr', threshold=3):
    """Detect outliers using various methods"""
    try:
        outliers = pd.Series(False, index=data.index)
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers.loc[data.dropna().index] = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
            
        return outliers, None
        
    except Exception as e:
        return None, f"Error detecting outliers: {str(e)}"

def handle_outliers(data, outliers, method='replace_median'):
    """Handle outliers using various methods"""
    try:
        cleaned_data = data.copy()
        
        if method == 'replace_median':
            median_value = data.median()
            cleaned_data.loc[outliers] = median_value
            
        elif method == 'replace_mean':
            mean_value = data.mean()
            cleaned_data.loc[outliers] = mean_value
            
        elif method == 'remove':
            cleaned_data = cleaned_data.loc[~outliers]
            
        elif method == 'cap':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = cleaned_data.clip(lower=lower_bound, upper=upper_bound)
            
        return cleaned_data, None
        
    except Exception as e:
        return None, f"Error handling outliers: {str(e)}"

def detect_missing_values(data):
    """Detect missing values and provide statistics"""
    try:
        missing_count = data.isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100
        
        missing_info = {
            'total_missing': missing_count.sum(),
            'percentage': missing_percentage,
            'missing_by_column': missing_count,
            'has_missing': missing_count.sum() > 0
        }
        
        return missing_info, None
        
    except Exception as e:
        return None, f"Error detecting missing values: {str(e)}"

def interpolate_missing_values(data, method='linear'):
    """Interpolate missing values using various methods"""
    try:
        interpolated_data = data.copy()
        
        if method == 'linear':
            interpolated_data = interpolated_data.interpolate(method='linear')
            
        elif method == 'polynomial':
            interpolated_data = interpolated_data.interpolate(method='polynomial', order=2)
            
        elif method == 'spline':
            interpolated_data = interpolated_data.interpolate(method='spline', order=3)
            
        elif method == 'forward_fill':
            interpolated_data = interpolated_data.fillna(method='ffill')
            
        elif method == 'backward_fill':
            interpolated_data = interpolated_data.fillna(method='bfill')
            
        elif method == 'mean':
            interpolated_data = interpolated_data.fillna(data.mean())
            
        elif method == 'median':
            interpolated_data = interpolated_data.fillna(data.median())
            
        # Fill any remaining NaN values with forward fill
        interpolated_data = interpolated_data.fillna(method='ffill').fillna(method='bfill')
        
        return interpolated_data, None
        
    except Exception as e:
        return None, f"Error interpolating missing values: {str(e)}"

def normalize_data(data, method='minmax'):
    """Normalize or standardize data using various methods"""
    try:
        if method == 'minmax':
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(data.values.reshape(-1, 1))
            normalized_data = pd.Series(scaled_values.flatten(), index=data.index, name=data.name)
            
        elif method == 'zscore':
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(data.values.reshape(-1, 1))
            normalized_data = pd.Series(scaled_values.flatten(), index=data.index, name=data.name)
            
        elif method == 'robust':
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(data.values.reshape(-1, 1))
            normalized_data = pd.Series(scaled_values.flatten(), index=data.index, name=data.name)
            
        elif method == 'log':
            # Add small constant to handle zero values
            min_val = data.min()
            if min_val <= 0:
                data_shifted = data + abs(min_val) + 1
            else:
                data_shifted = data
            normalized_data = np.log(data_shifted)
            
        else:
            return None, f"Unknown normalization method: {method}"
            
        return normalized_data, scaler if method != 'log' else None, None
        
    except Exception as e:
        return None, None, f"Error normalizing data: {str(e)}"

def check_data_quality(data):
    """Comprehensive data quality check"""
    try:
        quality_report = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_dates': data.index.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'date_range': {
                'start': data.index.min(),
                'end': data.index.max(),
                'span_days': (data.index.max() - data.index.min()).days
            },
            'numeric_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Check for irregular time intervals
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            quality_report['time_intervals'] = {
                'regular': len(time_diffs.unique()) == 1,
                'min_interval': time_diffs.min(),
                'max_interval': time_diffs.max(),
                'median_interval': time_diffs.median()
            }
        
        return quality_report, None
        
    except Exception as e:
        return None, f"Error checking data quality: {str(e)}"

def suggest_preprocessing_steps(data):
    """Suggest preprocessing steps based on data analysis"""
    try:
        suggestions = []
        
        # Check for missing values
        missing_info, _ = detect_missing_values(data)
        if missing_info and missing_info['has_missing']:
            suggestions.append({
                'step': 'Handle Missing Values',
                'reason': f"Found {missing_info['total_missing']} missing values",
                'recommended_method': 'linear interpolation'
            })
        
        # Check for outliers
        for col in data.select_dtypes(include=[np.number]).columns:
            outliers, _ = detect_outliers(data[col])
            if outliers is not None and outliers.sum() > 0:
                suggestions.append({
                    'step': 'Handle Outliers',
                    'reason': f"Found {outliers.sum()} outliers in {col}",
                    'recommended_method': 'replace with median'
                })
        
        # Check data scale
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].std() > 1000 or data[col].max() > 10000:
                suggestions.append({
                    'step': 'Normalize Data',
                    'reason': f"Large scale values in {col}",
                    'recommended_method': 'z-score normalization'
                })
        
        return suggestions, None
        
    except Exception as e:
        return None, f"Error suggesting preprocessing steps: {str(e)}"

def apply_preprocessing_pipeline(data, steps):
    """Apply a series of preprocessing steps"""
    try:
        processed_data = data.copy()
        processing_log = []
        
        for step in steps:
            if step['action'] == 'handle_outliers':
                outliers, _ = detect_outliers(processed_data.iloc[:, 0], method=step['outlier_method'])
                if outliers is not None and outliers.sum() > 0:
                    processed_data.iloc[:, 0], _ = handle_outliers(processed_data.iloc[:, 0], outliers, method=step['outlier_treatment'])
                    processing_log.append(f"Handled {outliers.sum()} outliers using {step['outlier_treatment']}")
            
            elif step['action'] == 'interpolate_missing':
                if processed_data.isnull().sum().sum() > 0:
                    processed_data, _ = interpolate_missing_values(processed_data, method=step['interpolation_method'])
                    processing_log.append(f"Interpolated missing values using {step['interpolation_method']}")
            
            elif step['action'] == 'normalize':
                processed_data.iloc[:, 0], _, _ = normalize_data(processed_data.iloc[:, 0], method=step['normalization_method'])
                processing_log.append(f"Normalized data using {step['normalization_method']}")
        
        return processed_data, processing_log, None
        
    except Exception as e:
        return None, None, f"Error applying preprocessing pipeline: {str(e)}"
