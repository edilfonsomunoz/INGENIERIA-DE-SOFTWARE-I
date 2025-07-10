import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def fit_ar_model(data, lags=None, max_lags=10):
    """
    Fit AutoRegressive (AR) model
    """
    try:
        # Determine optimal lags if not provided
        if lags is None:
            # Use AIC to select optimal lags
            aic_values = []
            for p in range(1, max_lags + 1):
                try:
                    model = AutoReg(data, lags=p)
                    fitted = model.fit()
                    aic_values.append(fitted.aic)
                except:
                    aic_values.append(np.inf)
            
            lags = np.argmin(aic_values) + 1
        
        # Fit AR model
        model = AutoReg(data, lags=lags)
        fitted_model = model.fit()
        
        # Get fitted values and residuals
        fitted_values = fitted_model.fittedvalues
        residuals = fitted_model.resid
        
        result = {
            'model': fitted_model,
            'model_type': 'AR',
            'lags': lags,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'fitted_values': fitted_values,
            'residuals': residuals,
            'params': fitted_model.params,
            'summary': fitted_model.summary()
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

def fit_arima_model(data, order=(1, 1, 1)):
    """
    Fit ARIMA model
    """
    try:
        # Fit ARIMA model
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        
        # Get fitted values and residuals
        fitted_values = fitted_model.fittedvalues
        residuals = fitted_model.resid
        
        result = {
            'model': fitted_model,
            'model_type': 'ARIMA',
            'order': order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'fitted_values': fitted_values,
            'residuals': residuals,
            'params': fitted_model.params,
            'summary': fitted_model.summary()
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

def auto_arima_selection(data, max_p=5, max_d=2, max_q=5):
    """
    Automatically select best ARIMA parameters
    """
    try:
        best_aic = np.inf
        best_order = None
        best_model = None
        
        with st.spinner("Buscando mejores parámetros ARIMA..."):
            progress_bar = st.progress(0)
            total_iterations = (max_p + 1) * (max_d + 1) * (max_q + 1)
            current_iteration = 0
            
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            fitted = model.fit()
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_model = fitted
                                
                        except:
                            continue
                        
                        current_iteration += 1
                        progress_bar.progress(current_iteration / total_iterations)
            
            progress_bar.empty()
        
        if best_model is None:
            return None, "No se pudo encontrar un modelo ARIMA adecuado"
        
        result = {
            'model': best_model,
            'model_type': 'ARIMA',
            'order': best_order,
            'aic': best_model.aic,
            'bic': best_model.bic,
            'fitted_values': best_model.fittedvalues,
            'residuals': best_model.resid,
            'params': best_model.params,
            'summary': best_model.summary()
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

def fit_garch_model(data, vol='GARCH', p=1, q=1):
    """
    Fit GARCH model for volatility modeling
    """
    try:
        # Prepare data (ensure it's a pandas Series)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        
        # Calculate returns if data appears to be prices
        if data.min() > 0 and data.std() / data.mean() < 0.1:
            # Likely price data, calculate returns
            returns = data.pct_change().dropna() * 100
        else:
            returns = data.dropna()
        
        # Fit GARCH model
        model = arch_model(returns, vol=vol, p=p, q=q)
        fitted_model = model.fit(disp='off')
        
        # Get conditional volatility
        conditional_volatility = fitted_model.conditional_volatility
        
        result = {
            'model': fitted_model,
            'model_type': 'GARCH',
            'vol_type': vol,
            'p': p,
            'q': q,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'conditional_volatility': conditional_volatility,
            'residuals': fitted_model.resid,
            'params': fitted_model.params,
            'summary': fitted_model.summary(),
            'returns': returns
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

def fit_egarch_model(data, p=1, o=1, q=1):
    """
    Fit EGARCH model for asymmetric volatility modeling
    """
    try:
        # Prepare data (ensure it's a pandas Series)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        
        # Calculate returns if data appears to be prices
        if data.min() > 0 and data.std() / data.mean() < 0.1:
            # Likely price data, calculate returns
            returns = data.pct_change().dropna() * 100
        else:
            returns = data.dropna()
        
        # Fit EGARCH model
        model = arch_model(returns, vol='EGARCH', p=p, o=o, q=q)
        fitted_model = model.fit(disp='off')
        
        # Get conditional volatility
        conditional_volatility = fitted_model.conditional_volatility
        
        result = {
            'model': fitted_model,
            'model_type': 'EGARCH',
            'p': p,
            'o': o,
            'q': q,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'conditional_volatility': conditional_volatility,
            'residuals': fitted_model.resid,
            'params': fitted_model.params,
            'summary': fitted_model.summary(),
            'returns': returns
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

def validate_model(model_result, data):
    """
    Validate model performance
    """
    try:
        model_type = model_result['model_type']
        
        if model_type in ['AR', 'ARIMA']:
            # For AR/ARIMA models
            fitted_values = model_result['fitted_values']
            residuals = model_result['residuals']
            
            # Align data for comparison
            if len(fitted_values) < len(data):
                actual_values = data.iloc[-len(fitted_values):]
            else:
                actual_values = data
            
            # Calculate metrics
            mse = mean_squared_error(actual_values, fitted_values)
            mae = mean_absolute_error(actual_values, fitted_values)
            rmse = np.sqrt(mse)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((actual_values - fitted_values) / actual_values)) * 100
            
            validation_result = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'residuals_mean': residuals.mean(),
                'residuals_std': residuals.std(),
                'ljung_box_p': None,  # Could add Ljung-Box test
                'jarque_bera_p': None  # Could add Jarque-Bera test
            }
            
        elif model_type in ['GARCH', 'EGARCH']:
            # For GARCH/EGARCH models
            returns = model_result['returns']
            conditional_volatility = model_result['conditional_volatility']
            residuals = model_result['residuals']
            
            # Calculate standardized residuals
            std_residuals = residuals / conditional_volatility
            
            validation_result = {
                'log_likelihood': model_result['model'].loglikelihood,
                'aic': model_result['aic'],
                'bic': model_result['bic'],
                'std_residuals_mean': std_residuals.mean(),
                'std_residuals_std': std_residuals.std(),
                'volatility_mean': conditional_volatility.mean(),
                'volatility_std': conditional_volatility.std()
            }
        
        return validation_result, None
        
    except Exception as e:
        return None, str(e)

def compare_models(model_results):
    """
    Compare multiple models based on information criteria
    """
    try:
        comparison_data = []
        
        for name, model_result in model_results.items():
            comparison_data.append({
                'Model': name,
                'Type': model_result['model_type'],
                'AIC': model_result['aic'],
                'BIC': model_result['bic'],
                'Parameters': len(model_result['params'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by AIC
        comparison_df = comparison_df.sort_values('AIC')
        comparison_df['AIC_Rank'] = range(1, len(comparison_df) + 1)
        
        # Rank models by BIC
        comparison_df = comparison_df.sort_values('BIC')
        comparison_df['BIC_Rank'] = range(1, len(comparison_df) + 1)
        
        # Sort by AIC for final display
        comparison_df = comparison_df.sort_values('AIC')
        
        return comparison_df, None
        
    except Exception as e:
        return None, str(e)

def generate_model_forecasts(model_result, steps=12):
    """
    Generate forecasts from fitted models
    """
    try:
        model_type = model_result['model_type']
        
        if model_type == 'AR':
            # AR forecasts
            forecasts = model_result['model'].forecast(steps=steps)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'forecast': forecasts,
                'lower_ci': forecasts - 1.96 * np.sqrt(model_result['model'].sigma2),
                'upper_ci': forecasts + 1.96 * np.sqrt(model_result['model'].sigma2)
            })
            
        elif model_type == 'ARIMA':
            # ARIMA forecasts
            forecast_result = model_result['model'].forecast(steps=steps)
            conf_int = model_result['model'].get_forecast(steps=steps).conf_int()
            
            forecast_df = pd.DataFrame({
                'forecast': forecast_result,
                'lower_ci': conf_int.iloc[:, 0],
                'upper_ci': conf_int.iloc[:, 1]
            })
            
        elif model_type in ['GARCH', 'EGARCH']:
            # GARCH/EGARCH forecasts (volatility forecasts)
            volatility_forecast = model_result['model'].forecast(horizon=steps)
            
            forecast_df = pd.DataFrame({
                'volatility_forecast': volatility_forecast.variance.iloc[-1, :],
                'volatility_lower_ci': volatility_forecast.variance.iloc[-1, :] * 0.8,
                'volatility_upper_ci': volatility_forecast.variance.iloc[-1, :] * 1.2
            })
        
        return forecast_df, None
        
    except Exception as e:
        return None, str(e)