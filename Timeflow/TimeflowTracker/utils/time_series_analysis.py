import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import jarque_bera
import warnings
warnings.filterwarnings('ignore')

def decompose_time_series(data, model='additive', period=None):
    """Decompose time series into trend, seasonal, and residual components"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        
        # Auto-detect period if not provided
        if period is None:
            # Try to infer frequency from data
            if len(series) >= 24:
                period = 12  # Default to 12 for monthly-like data
            elif len(series) >= 14:
                period = 7   # Weekly pattern
            else:
                period = 4   # Quarterly pattern
        
        # Perform decomposition
        decomposition = seasonal_decompose(series, model=model, period=period)
        
        result = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'model': model,
            'period': period
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error decomposing time series: {str(e)}"

def test_stationarity(data, test_type='adf'):
    """Test for stationarity using ADF or KPSS test"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        
        # Remove NaN values
        series = series.dropna()
        
        if test_type == 'adf':
            # Augmented Dickey-Fuller test
            result = adfuller(series)
            
            test_result = {
                'test_name': 'Augmented Dickey-Fuller Test',
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] <= 0.05,
                'interpretation': 'Stationary' if result[1] <= 0.05 else 'Non-stationary'
            }
            
        elif test_type == 'kpss':
            # KPSS test
            result = kpss(series)
            
            test_result = {
                'test_name': 'KPSS Test',
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[3],
                'is_stationary': result[1] >= 0.05,
                'interpretation': 'Stationary' if result[1] >= 0.05 else 'Non-stationary'
            }
        
        return test_result, None
        
    except Exception as e:
        return None, f"Error testing stationarity: {str(e)}"

def calculate_acf_pacf(data, lags=40):
    """Calculate ACF and PACF for time series"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        
        # Remove NaN values
        series = series.dropna()
        
        # Ensure we don't have more lags than data points
        max_lags = min(lags, len(series) - 1)
        
        # Calculate ACF and PACF
        acf_values = acf(series, nlags=max_lags)
        pacf_values = pacf(series, nlags=max_lags)
        
        result = {
            'acf': acf_values,
            'pacf': pacf_values,
            'lags': range(max_lags + 1)
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error calculating ACF/PACF: {str(e)}"

def identify_temporal_patterns(data):
    """Identify temporal patterns in the time series"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        series = series.dropna()
        
        patterns = {}
        
        # Test for stationarity
        adf_result, _ = test_stationarity(data, 'adf')
        kpss_result, _ = test_stationarity(data, 'kpss')
        
        patterns['stationarity'] = {
            'adf_test': adf_result,
            'kpss_test': kpss_result,
            'likely_stationary': adf_result['is_stationary'] and kpss_result['is_stationary']
        }
        
        # Decompose to identify trend and seasonality
        decomposition, _ = decompose_time_series(data)
        
        if decomposition:
            # Check for trend
            trend_component = decomposition['trend'].dropna()
            if len(trend_component) > 0:
                trend_slope = np.polyfit(range(len(trend_component)), trend_component, 1)[0]
                patterns['trend'] = {
                    'present': abs(trend_slope) > 0.01,
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'slope': trend_slope
                }
            
            # Check for seasonality
            seasonal_component = decomposition['seasonal'].dropna()
            if len(seasonal_component) > 0:
                seasonal_variance = seasonal_component.var()
                patterns['seasonality'] = {
                    'present': seasonal_variance > 0.1,
                    'strength': seasonal_variance,
                    'period': decomposition['period']
                }
        
        # Calculate ACF/PACF for pattern identification
        acf_pacf, _ = calculate_acf_pacf(data)
        if acf_pacf:
            patterns['autocorrelation'] = {
                'significant_acf_lags': np.where(np.abs(acf_pacf['acf'][1:]) > 0.2)[0] + 1,
                'significant_pacf_lags': np.where(np.abs(acf_pacf['pacf'][1:]) > 0.2)[0] + 1
            }
        
        # Basic statistics
        patterns['statistics'] = {
            'mean': series.mean(),
            'std': series.std(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'jarque_bera_test': jarque_bera(series)
        }
        
        return patterns, None
        
    except Exception as e:
        return None, f"Error identifying temporal patterns: {str(e)}"

def recommend_model(patterns):
    """Recommend appropriate time series model based on patterns"""
    try:
        recommendations = []
        
        # Check stationarity
        if patterns.get('stationarity', {}).get('likely_stationary', False):
            recommendations.append({
                'model': 'ARMA',
                'reason': 'Data appears to be stationary',
                'parameters': 'Start with ARMA(1,1) and adjust based on ACF/PACF'
            })
        else:
            recommendations.append({
                'model': 'ARIMA',
                'reason': 'Data appears to be non-stationary, differencing may be needed',
                'parameters': 'Consider ARIMA(1,1,1) as starting point'
            })
        
        # Check for seasonality
        if patterns.get('seasonality', {}).get('present', False):
            recommendations.append({
                'model': 'SARIMA',
                'reason': 'Seasonal patterns detected',
                'parameters': f"Include seasonal component with period {patterns['seasonality']['period']}"
            })
        
        # Check for trend
        if patterns.get('trend', {}).get('present', False):
            recommendations.append({
                'model': 'ARIMA with trend',
                'reason': f"Trend detected ({patterns['trend']['direction']})",
                'parameters': 'Consider including trend component in the model'
            })
        
        # Default recommendation
        if not recommendations:
            recommendations.append({
                'model': 'ARIMA',
                'reason': 'General purpose model suitable for most time series',
                'parameters': 'Start with ARIMA(1,1,1) and adjust based on diagnostics'
            })
        
        return recommendations, None
        
    except Exception as e:
        return None, f"Error recommending model: {str(e)}"

def difference_series(data, differences=1):
    """Apply differencing to make series stationary"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        
        differenced = series.copy()
        
        for i in range(differences):
            differenced = differenced.diff().dropna()
        
        return differenced, None
        
    except Exception as e:
        return None, f"Error differencing series: {str(e)}"

def seasonal_difference(data, period=12):
    """Apply seasonal differencing"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        
        # Apply seasonal differencing
        differenced = series.diff(period).dropna()
        
        return differenced, None
        
    except Exception as e:
        return None, f"Error applying seasonal differencing: {str(e)}"

def detect_changepoints(data, threshold=0.1):
    """Detect structural changes in the time series"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        series = series.dropna()
        
        # Simple changepoint detection using rolling statistics
        window = max(10, len(series) // 10)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Detect points where rolling statistics change significantly
        mean_changes = rolling_mean.diff().abs() > threshold * series.std()
        std_changes = rolling_std.diff().abs() > threshold * series.std()
        
        changepoints = series.index[mean_changes | std_changes].tolist()
        
        return changepoints, None
        
    except Exception as e:
        return None, f"Error detecting changepoints: {str(e)}"
