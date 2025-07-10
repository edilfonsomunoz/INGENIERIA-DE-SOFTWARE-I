import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

def generate_forecasts(model_result, steps=12):
    """Generate forecasts using fitted SARIMA model"""
    try:
        model = model_result['model']
        
        # Generate forecasts
        forecast_result = model.forecast(steps=steps)
        
        # Get confidence intervals
        forecast_ci = model.get_forecast(steps=steps).conf_int()
        
        # Create forecast dates
        last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else pd.Timestamp.now()
        
        if hasattr(model.data, 'freq') and model.data.freq:
            freq = model.data.freq
        else:
            # Infer frequency from existing data
            if len(model.data.dates) > 1:
                freq = pd.infer_freq(model.data.dates)
            else:
                freq = 'D'  # Default to daily
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=steps, freq=freq)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'forecast': forecast_result,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        }, index=forecast_dates)
        
        result = {
            'forecasts': forecast_df,
            'forecast_values': forecast_result,
            'confidence_intervals': {
                'lower': forecast_ci.iloc[:, 0],
                'upper': forecast_ci.iloc[:, 1]
            },
            'forecast_dates': forecast_dates,
            'steps': steps
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error generating forecasts: {str(e)}"

def calculate_forecast_metrics(actual, predicted):
    """Calculate forecast accuracy metrics"""
    try:
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return None, "No valid data points for metric calculation"
        
        # Calculate metrics
        mse = np.mean((actual_clean - predicted_clean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_clean - predicted_clean))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        
        # Mean Absolute Scaled Error (MASE)
        naive_errors = np.abs(np.diff(actual_clean))
        mase = mae / np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) != 0 else np.inf
        
        # Theil's U statistic
        theil_u = np.sqrt(np.mean((predicted_clean - actual_clean) ** 2)) / \
                  (np.sqrt(np.mean(predicted_clean ** 2)) + np.sqrt(np.mean(actual_clean ** 2)))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mase': mase,
            'theil_u': theil_u,
            'r_squared': np.corrcoef(actual_clean, predicted_clean)[0, 1] ** 2
        }
        
        return metrics, None
        
    except Exception as e:
        return None, f"Error calculating forecast metrics: {str(e)}"

def backtest_model(data, model_order, seasonal_order, test_size=0.2):
    """Perform backtesting on the model"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        series = series.dropna()
        
        # Split data
        split_point = int(len(series) * (1 - test_size))
        train_data = series[:split_point]
        test_data = series[split_point:]
        
        # Fit model on training data
        model = SARIMAX(train_data, order=model_order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Generate forecasts for test period
        forecast_steps = len(test_data)
        forecasts = fitted_model.forecast(steps=forecast_steps)
        
        # Calculate metrics
        metrics, error = calculate_forecast_metrics(test_data.values, forecasts.values)
        
        if error:
            return None, error
        
        result = {
            'train_data': train_data,
            'test_data': test_data,
            'forecasts': forecasts,
            'metrics': metrics,
            'model': fitted_model
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error in backtesting: {str(e)}"

def cross_validate_model(data, model_order, seasonal_order, n_splits=5):
    """Perform time series cross-validation"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        series = series.dropna()
        
        # Calculate split sizes
        total_length = len(series)
        test_size = total_length // (n_splits + 1)
        
        cv_results = []
        
        for i in range(n_splits):
            # Define train and test splits
            train_end = total_length - (n_splits - i) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            train_data = series[:train_end]
            test_data = series[test_start:test_end]
            
            if len(train_data) < 20 or len(test_data) < 1:
                continue
            
            try:
                # Fit model
                model = SARIMAX(train_data, order=model_order, seasonal_order=seasonal_order)
                fitted_model = model.fit(disp=False)
                
                # Generate forecasts
                forecasts = fitted_model.forecast(steps=len(test_data))
                
                # Calculate metrics
                metrics, _ = calculate_forecast_metrics(test_data.values, forecasts.values)
                
                if metrics:
                    cv_results.append({
                        'split': i + 1,
                        'train_size': len(train_data),
                        'test_size': len(test_data),
                        'metrics': metrics
                    })
                    
            except:
                continue
        
        if not cv_results:
            return None, "Cross-validation failed - no valid splits"
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in cv_results[0]['metrics'].keys():
            avg_metrics[metric] = np.mean([result['metrics'][metric] for result in cv_results])
            avg_metrics[f'{metric}_std'] = np.std([result['metrics'][metric] for result in cv_results])
        
        result = {
            'cv_results': cv_results,
            'average_metrics': avg_metrics,
            'n_splits': len(cv_results)
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error in cross-validation: {str(e)}"

def forecast_scenarios(model_result, steps=12, scenarios=['optimistic', 'pessimistic', 'realistic']):
    """Generate different forecast scenarios"""
    try:
        base_forecast, error = generate_forecasts(model_result, steps)
        
        if error:
            return None, error
        
        scenarios_data = {}
        
        # Base forecast (realistic scenario)
        scenarios_data['realistic'] = base_forecast['forecasts']['forecast']
        
        # Optimistic scenario (upper confidence interval)
        scenarios_data['optimistic'] = base_forecast['confidence_intervals']['upper']
        
        # Pessimistic scenario (lower confidence interval)
        scenarios_data['pessimistic'] = base_forecast['confidence_intervals']['lower']
        
        # Create DataFrame
        scenarios_df = pd.DataFrame(scenarios_data, index=base_forecast['forecast_dates'])
        
        result = {
            'scenarios': scenarios_df,
            'base_forecast': base_forecast,
            'scenario_names': scenarios
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error generating forecast scenarios: {str(e)}"

def export_forecasts(forecast_result, filename='forecasts.csv'):
    """Export forecasts to CSV file"""
    try:
        forecast_df = forecast_result['forecasts']
        
        # Add metadata
        forecast_df['forecast_date'] = pd.Timestamp.now()
        forecast_df['model_type'] = 'SARIMA'
        
        # Save to CSV
        csv_data = forecast_df.to_csv()
        
        return csv_data, None
        
    except Exception as e:
        return None, f"Error exporting forecasts: {str(e)}"

def calculate_forecast_intervals(model_result, steps=12, confidence_levels=[0.80, 0.90, 0.95]):
    """Calculate multiple confidence intervals for forecasts"""
    try:
        model = model_result['model']
        
        # Get forecast with different confidence levels
        intervals = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            forecast_obj = model.get_forecast(steps=steps, alpha=alpha)
            
            intervals[f'{int(confidence*100)}%'] = {
                'lower': forecast_obj.conf_int().iloc[:, 0],
                'upper': forecast_obj.conf_int().iloc[:, 1]
            }
        
        # Generate forecast dates
        last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else pd.Timestamp.now()
        
        if hasattr(model.data, 'freq') and model.data.freq:
            freq = model.data.freq
        else:
            freq = 'D'
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=steps, freq=freq)
        
        # Get point forecasts
        point_forecasts = model.forecast(steps=steps)
        
        result = {
            'point_forecasts': point_forecasts,
            'confidence_intervals': intervals,
            'forecast_dates': forecast_dates,
            'confidence_levels': confidence_levels
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error calculating forecast intervals: {str(e)}"

def forecast_diagnostics(forecast_result, actual_data=None):
    """Perform diagnostics on forecast results"""
    try:
        diagnostics = {}
        
        # Basic forecast statistics
        forecasts = forecast_result['forecast_values']
        diagnostics['forecast_stats'] = {
            'mean': forecasts.mean(),
            'std': forecasts.std(),
            'min': forecasts.min(),
            'max': forecasts.max(),
            'trend': 'increasing' if forecasts.iloc[-1] > forecasts.iloc[0] else 'decreasing'
        }
        
        # Confidence interval width analysis
        if 'confidence_intervals' in forecast_result:
            ci_width = forecast_result['confidence_intervals']['upper'] - forecast_result['confidence_intervals']['lower']
            diagnostics['uncertainty'] = {
                'avg_ci_width': ci_width.mean(),
                'max_ci_width': ci_width.max(),
                'min_ci_width': ci_width.min(),
                'increasing_uncertainty': ci_width.iloc[-1] > ci_width.iloc[0]
            }
        
        # If actual data is provided, calculate accuracy
        if actual_data is not None:
            metrics, _ = calculate_forecast_metrics(actual_data, forecasts)
            if metrics:
                diagnostics['accuracy'] = metrics
        
        return diagnostics, None
        
    except Exception as e:
        return None, f"Error in forecast diagnostics: {str(e)}"
