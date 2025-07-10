import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import normaltest
import warnings
warnings.filterwarnings('ignore')

def fit_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Fit SARIMA model to time series data"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        series = series.dropna()
        
        # Fit SARIMA model
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Calculate metrics
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        result = {
            'model': fitted_model,
            'aic': aic,
            'bic': bic,
            'order': order,
            'seasonal_order': seasonal_order,
            'summary': fitted_model.summary(),
            'fitted_values': fitted_model.fittedvalues,
            'residuals': fitted_model.resid
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error fitting SARIMA model: {str(e)}"

def validate_sarima_model(model_result):
    """Validate SARIMA model using residual analysis"""
    try:
        residuals = model_result['residuals']
        
        validation_results = {}
        
        # Ljung-Box test for autocorrelation in residuals
        try:
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            validation_results['ljung_box'] = {
                'test_statistic': ljung_box['lb_stat'].iloc[-1],
                'p_value': ljung_box['lb_pvalue'].iloc[-1],
                'passed': ljung_box['lb_pvalue'].iloc[-1] > 0.05
            }
        except:
            validation_results['ljung_box'] = {'passed': False, 'error': 'Could not perform Ljung-Box test'}
        
        # Jarque-Bera test for normality of residuals
        try:
            jb_stat, jb_pvalue = jarque_bera(residuals.dropna())
            validation_results['jarque_bera'] = {
                'test_statistic': jb_stat,
                'p_value': jb_pvalue,
                'passed': jb_pvalue > 0.05
            }
        except:
            validation_results['jarque_bera'] = {'passed': False, 'error': 'Could not perform Jarque-Bera test'}
        
        # Shapiro-Wilk test for normality (alternative)
        try:
            from scipy.stats import shapiro
            if len(residuals) <= 5000:  # Shapiro-Wilk is limited to 5000 samples
                sw_stat, sw_pvalue = shapiro(residuals.dropna())
                validation_results['shapiro_wilk'] = {
                    'test_statistic': sw_stat,
                    'p_value': sw_pvalue,
                    'passed': sw_pvalue > 0.05
                }
        except:
            pass
        
        # Basic residual statistics
        validation_results['residual_stats'] = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
            'min': residuals.min(),
            'max': residuals.max()
        }
        
        # Overall model validation
        passed_tests = sum([test.get('passed', False) for test in validation_results.values() if isinstance(test, dict) and 'passed' in test])
        total_tests = len([test for test in validation_results.values() if isinstance(test, dict) and 'passed' in test])
        
        validation_results['overall'] = {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'validation_score': passed_tests / total_tests if total_tests > 0 else 0,
            'is_valid': passed_tests >= total_tests * 0.7  # 70% threshold
        }
        
        return validation_results, None
        
    except Exception as e:
        return None, f"Error validating SARIMA model: {str(e)}"

def auto_arima_selection(data, max_p=3, max_d=2, max_q=3, max_P=2, max_D=1, max_Q=2, m=12):
    """Automatically select best ARIMA parameters using grid search"""
    try:
        # Get the series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        series = series.dropna()
        
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        best_model = None
        
        results = []
        
        # Grid search
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    for P in range(max_P + 1):
                        for D in range(max_D + 1):
                            for Q in range(max_Q + 1):
                                try:
                                    model = SARIMAX(series, 
                                                   order=(p, d, q),
                                                   seasonal_order=(P, D, Q, m))
                                    fitted_model = model.fit(disp=False)
                                    
                                    aic = fitted_model.aic
                                    bic = fitted_model.bic
                                    
                                    results.append({
                                        'order': (p, d, q),
                                        'seasonal_order': (P, D, Q, m),
                                        'aic': aic,
                                        'bic': bic,
                                        'model': fitted_model
                                    })
                                    
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, m)
                                        best_model = fitted_model
                                        
                                except:
                                    continue
        
        if best_model is None:
            return None, "Could not find suitable ARIMA parameters"
        
        # Sort results by AIC
        results.sort(key=lambda x: x['aic'])
        
        selection_result = {
            'best_model': best_model,
            'best_order': best_order,
            'best_seasonal_order': best_seasonal_order,
            'best_aic': best_aic,
            'all_results': results[:10],  # Top 10 models
            'fitted_values': best_model.fittedvalues,
            'residuals': best_model.resid
        }
        
        return selection_result, None
        
    except Exception as e:
        return None, f"Error in auto ARIMA selection: {str(e)}"

def diagnose_model_fit(model_result):
    """Diagnose model fit and provide recommendations"""
    try:
        residuals = model_result['residuals']
        
        diagnostics = {}
        
        # Residual analysis
        diagnostics['residual_analysis'] = {
            'mean_close_to_zero': abs(residuals.mean()) < 0.1,
            'constant_variance': residuals.std() / residuals.mean() < 2 if residuals.mean() != 0 else True,
            'no_patterns': True  # This would require more sophisticated analysis
        }
        
        # Model fit metrics
        diagnostics['fit_metrics'] = {
            'aic': model_result['aic'],
            'bic': model_result['bic'],
            'log_likelihood': model_result['model'].llf if hasattr(model_result['model'], 'llf') else None
        }
        
        # Parameter significance
        try:
            params = model_result['model'].params
            pvalues = model_result['model'].pvalues
            
            significant_params = sum(pvalues < 0.05)
            total_params = len(params)
            
            diagnostics['parameters'] = {
                'significant_parameters': significant_params,
                'total_parameters': total_params,
                'significance_ratio': significant_params / total_params
            }
        except:
            diagnostics['parameters'] = {'error': 'Could not analyze parameters'}
        
        # Recommendations
        recommendations = []
        
        if diagnostics['residual_analysis']['mean_close_to_zero']:
            recommendations.append("✓ Residuals have mean close to zero")
        else:
            recommendations.append("⚠ Residuals may have non-zero mean - consider model adjustment")
        
        if diagnostics['fit_metrics']['aic'] is not None:
            recommendations.append(f"Model AIC: {diagnostics['fit_metrics']['aic']:.2f}")
        
        if diagnostics['parameters'].get('significance_ratio', 0) > 0.7:
            recommendations.append("✓ Most parameters are statistically significant")
        else:
            recommendations.append("⚠ Some parameters may not be significant")
        
        diagnostics['recommendations'] = recommendations
        
        return diagnostics, None
        
    except Exception as e:
        return None, f"Error diagnosing model fit: {str(e)}"

def compare_models(model_results):
    """Compare multiple SARIMA models"""
    try:
        comparison = []
        
        for i, model_result in enumerate(model_results):
            comparison.append({
                'model_id': i + 1,
                'order': model_result['order'],
                'seasonal_order': model_result['seasonal_order'],
                'aic': model_result['aic'],
                'bic': model_result['bic'],
                'model': model_result['model']
            })
        
        # Sort by AIC
        comparison.sort(key=lambda x: x['aic'])
        
        # Add rankings
        for i, model in enumerate(comparison):
            model['aic_rank'] = i + 1
        
        # Sort by BIC and add BIC rankings
        comparison_bic = sorted(comparison, key=lambda x: x['bic'])
        for i, model in enumerate(comparison_bic):
            model['bic_rank'] = i + 1
        
        return comparison, None
        
    except Exception as e:
        return None, f"Error comparing models: {str(e)}"

def extract_model_components(model_result):
    """Extract and interpret model components"""
    try:
        model = model_result['model']
        
        components = {}
        
        # Parameter estimates
        if hasattr(model, 'params'):
            components['parameters'] = {
                'estimates': model.params.to_dict(),
                'standard_errors': model.bse.to_dict() if hasattr(model, 'bse') else {},
                'p_values': model.pvalues.to_dict() if hasattr(model, 'pvalues') else {}
            }
        
        # Model order
        components['order'] = {
            'non_seasonal': model_result['order'],
            'seasonal': model_result['seasonal_order']
        }
        
        # Fitted values and residuals
        components['fitted_values'] = model_result['fitted_values']
        components['residuals'] = model_result['residuals']
        
        # Model summary statistics
        components['statistics'] = {
            'aic': model_result['aic'],
            'bic': model_result['bic'],
            'log_likelihood': model.llf if hasattr(model, 'llf') else None,
            'scale': model.scale if hasattr(model, 'scale') else None
        }
        
        return components, None
        
    except Exception as e:
        return None, f"Error extracting model components: {str(e)}"

def generate_model_report(model_result, validation_result):
    """Generate a comprehensive model report"""
    try:
        report = {
            'model_specification': {
                'order': model_result['order'],
                'seasonal_order': model_result['seasonal_order'],
                'aic': model_result['aic'],
                'bic': model_result['bic']
            },
            'validation_summary': validation_result['overall'],
            'diagnostic_tests': {
                'ljung_box': validation_result.get('ljung_box', {}),
                'jarque_bera': validation_result.get('jarque_bera', {}),
                'shapiro_wilk': validation_result.get('shapiro_wilk', {})
            },
            'residual_statistics': validation_result.get('residual_stats', {}),
            'interpretation': []
        }
        
        # Generate interpretation
        if validation_result['overall']['is_valid']:
            report['interpretation'].append("✓ Model passes validation tests and appears suitable for forecasting")
        else:
            report['interpretation'].append("⚠ Model may have issues - consider parameter adjustment")
        
        if validation_result.get('ljung_box', {}).get('passed', False):
            report['interpretation'].append("✓ No significant autocorrelation in residuals")
        else:
            report['interpretation'].append("⚠ Residuals may have autocorrelation - model may be inadequate")
        
        if validation_result.get('jarque_bera', {}).get('passed', False):
            report['interpretation'].append("✓ Residuals appear normally distributed")
        else:
            report['interpretation'].append("⚠ Residuals may not be normally distributed")
        
        return report, None
        
    except Exception as e:
        return None, f"Error generating model report: {str(e)}"
