import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def generate_data_interpretation(data, data_info):
    """Generate interpretation of the loaded data"""
    try:
        interpretation = []
        
        # Basic data description
        interpretation.append(f"**Data Overview:**")
        interpretation.append(f"- Dataset contains {len(data)} observations")
        interpretation.append(f"- Time period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        interpretation.append(f"- Duration: {(data.index.max() - data.index.min()).days} days")
        
        # Data quality assessment
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            interpretation.append(f"- **Data Quality Issue:** {missing_count} missing values detected ({missing_count/len(data)*100:.1f}%)")
        else:
            interpretation.append("- **Data Quality:** No missing values detected")
        
        # Basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            main_col = numeric_cols[0]
            stats = data[main_col].describe()
            
            interpretation.append(f"\n**Statistical Summary:**")
            interpretation.append(f"- Mean: {stats['mean']:.2f}")
            interpretation.append(f"- Standard Deviation: {stats['std']:.2f}")
            interpretation.append(f"- Range: {stats['min']:.2f} to {stats['max']:.2f}")
            
            # Trend analysis
            if len(data) > 1:
                first_half = data[main_col][:len(data)//2].mean()
                second_half = data[main_col][len(data)//2:].mean()
                
                if second_half > first_half * 1.05:
                    interpretation.append("- **Trend:** Increasing trend observed")
                elif second_half < first_half * 0.95:
                    interpretation.append("- **Trend:** Decreasing trend observed")
                else:
                    interpretation.append("- **Trend:** Relatively stable over time")
        
        return "\n".join(interpretation), None
        
    except Exception as e:
        return None, f"Error generating data interpretation: {str(e)}"

def generate_analysis_interpretation(decomposition_result, patterns_result):
    """Generate interpretation of time series analysis results"""
    try:
        interpretation = []
        
        interpretation.append("**Time Series Analysis Results:**\n")
        
        # Stationarity interpretation
        if patterns_result and 'stationarity' in patterns_result:
            stationarity = patterns_result['stationarity']
            
            if stationarity.get('likely_stationary', False):
                interpretation.append("✅ **Stationarity:** The time series appears to be stationary, which is good for modeling.")
            else:
                interpretation.append("⚠️ **Stationarity:** The time series appears to be non-stationary. Consider differencing.")
            
            # ADF test interpretation
            adf_result = stationarity.get('adf_test', {})
            if adf_result:
                interpretation.append(f"- ADF Test p-value: {adf_result.get('p_value', 'N/A'):.4f}")
                interpretation.append(f"- ADF Test result: {adf_result.get('interpretation', 'N/A')}")
        
        # Trend interpretation
        if patterns_result and 'trend' in patterns_result:
            trend = patterns_result['trend']
            
            if trend.get('present', False):
                direction = trend.get('direction', 'unknown')
                interpretation.append(f"📈 **Trend:** {direction.capitalize()} trend detected")
            else:
                interpretation.append("📊 **Trend:** No significant trend detected")
        
        # Seasonality interpretation
        if patterns_result and 'seasonality' in patterns_result:
            seasonality = patterns_result['seasonality']
            
            if seasonality.get('present', False):
                period = seasonality.get('period', 'unknown')
                interpretation.append(f"🔄 **Seasonality:** Seasonal pattern detected with period {period}")
            else:
                interpretation.append("📋 **Seasonality:** No clear seasonal pattern detected")
        
        # Decomposition interpretation
        if decomposition_result:
            interpretation.append(f"\n**Decomposition Analysis:**")
            interpretation.append(f"- Model type: {decomposition_result.get('model', 'N/A')}")
            interpretation.append(f"- Seasonal period: {decomposition_result.get('period', 'N/A')}")
            interpretation.append("- The time series has been decomposed into trend, seasonal, and residual components")
        
        return "\n".join(interpretation), None
        
    except Exception as e:
        return None, f"Error generating analysis interpretation: {str(e)}"

def generate_model_interpretation(model_result, validation_result):
    """Generate interpretation of SARIMA model results"""
    try:
        interpretation = []
        
        interpretation.append("**SARIMA Model Results:**\n")
        
        # Model specification
        order = model_result.get('order', (0, 0, 0))
        seasonal_order = model_result.get('seasonal_order', (0, 0, 0, 0))
        
        interpretation.append(f"📋 **Model Specification:**")
        interpretation.append(f"- SARIMA{order}x{seasonal_order}")
        interpretation.append(f"- AIC: {model_result.get('aic', 'N/A'):.2f}")
        interpretation.append(f"- BIC: {model_result.get('bic', 'N/A'):.2f}")
        
        # Model validation
        if validation_result:
            overall_valid = validation_result.get('overall', {}).get('is_valid', False)
            
            if overall_valid:
                interpretation.append("\n✅ **Model Validation:** Model passes most diagnostic tests")
            else:
                interpretation.append("\n⚠️ **Model Validation:** Model may have some issues")
            
            # Specific tests
            ljung_box = validation_result.get('ljung_box', {})
            if ljung_box.get('passed', False):
                interpretation.append("- ✅ Ljung-Box test: No autocorrelation in residuals")
            else:
                interpretation.append("- ⚠️ Ljung-Box test: Possible autocorrelation in residuals")
            
            jarque_bera = validation_result.get('jarque_bera', {})
            if jarque_bera.get('passed', False):
                interpretation.append("- ✅ Jarque-Bera test: Residuals appear normally distributed")
            else:
                interpretation.append("- ⚠️ Jarque-Bera test: Residuals may not be normally distributed")
        
        # Model recommendations
        interpretation.append(f"\n**Recommendations:**")
        
        if validation_result and validation_result.get('overall', {}).get('is_valid', False):
            interpretation.append("- The model appears suitable for forecasting")
            interpretation.append("- Consider generating forecasts with confidence intervals")
        else:
            interpretation.append("- Consider adjusting model parameters")
            interpretation.append("- Try different SARIMA orders or additional preprocessing")
        
        return "\n".join(interpretation), None
        
    except Exception as e:
        return None, f"Error generating model interpretation: {str(e)}"

def generate_forecast_interpretation(forecast_result, forecast_diagnostics=None):
    """Generate interpretation of forecast results"""
    try:
        interpretation = []
        
        interpretation.append("**Forecast Results:**\n")
        
        # Basic forecast information
        steps = forecast_result.get('steps', 0)
        interpretation.append(f"📊 **Forecast Summary:**")
        interpretation.append(f"- Forecast horizon: {steps} periods")
        
        # Forecast statistics
        if forecast_diagnostics:
            forecast_stats = forecast_diagnostics.get('forecast_stats', {})
            
            mean_forecast = forecast_stats.get('mean', 0)
            trend = forecast_stats.get('trend', 'unknown')
            
            interpretation.append(f"- Average forecast value: {mean_forecast:.2f}")
            interpretation.append(f"- Forecast trend: {trend}")
            
            # Uncertainty analysis
            uncertainty = forecast_diagnostics.get('uncertainty', {})
            if uncertainty:
                avg_ci_width = uncertainty.get('avg_ci_width', 0)
                increasing_uncertainty = uncertainty.get('increasing_uncertainty', False)
                
                interpretation.append(f"- Average confidence interval width: {avg_ci_width:.2f}")
                
                if increasing_uncertainty:
                    interpretation.append("- ⚠️ Forecast uncertainty increases over time")
                else:
                    interpretation.append("- ✅ Forecast uncertainty remains relatively stable")
        
        # Forecast interpretation
        forecasts = forecast_result.get('forecast_values', pd.Series())
        if len(forecasts) > 0:
            interpretation.append(f"\n**Forecast Analysis:**")
            
            # Trend analysis
            if len(forecasts) > 1:
                if forecasts.iloc[-1] > forecasts.iloc[0]:
                    interpretation.append("- 📈 Forecasts show an upward trend")
                elif forecasts.iloc[-1] < forecasts.iloc[0]:
                    interpretation.append("- 📉 Forecasts show a downward trend")
                else:
                    interpretation.append("- 📊 Forecasts show a stable pattern")
            
            # Variability analysis
            forecast_std = forecasts.std()
            forecast_mean = forecasts.mean()
            
            if forecast_std / forecast_mean > 0.1:
                interpretation.append("- High variability in forecasts")
            else:
                interpretation.append("- Low variability in forecasts")
        
        # Confidence intervals
        if 'confidence_intervals' in forecast_result:
            interpretation.append(f"\n**Confidence Intervals:**")
            interpretation.append("- 95% confidence intervals provided")
            interpretation.append("- Wider intervals indicate higher uncertainty")
        
        # Recommendations
        interpretation.append(f"\n**Recommendations:**")
        interpretation.append("- Monitor actual values as they become available")
        interpretation.append("- Update the model periodically with new data")
        interpretation.append("- Consider forecast accuracy when making decisions")
        
        return "\n".join(interpretation), None
        
    except Exception as e:
        return None, f"Error generating forecast interpretation: {str(e)}"

def generate_comprehensive_report(data, analysis_results, model_results, forecast_results):
    """Generate a comprehensive analysis report"""
    try:
        report = []
        
        # Header
        report.append("# TIME FLOW - Time Series Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*60)
        
        # Data summary
        if data is not None:
            data_interp, _ = generate_data_interpretation(data, {})
            if data_interp:
                report.append("\n## 1. Data Overview")
                report.append(data_interp)
        
        # Analysis results
        if analysis_results:
            analysis_interp, _ = generate_analysis_interpretation(
                analysis_results.get('decomposition'),
                analysis_results.get('patterns')
            )
            if analysis_interp:
                report.append("\n## 2. Time Series Analysis")
                report.append(analysis_interp)
        
        # Model results
        if model_results:
            model_interp, _ = generate_model_interpretation(
                model_results.get('model'),
                model_results.get('validation')
            )
            if model_interp:
                report.append("\n## 3. Model Results")
                report.append(model_interp)
        
        # Forecast results
        if forecast_results:
            forecast_interp, _ = generate_forecast_interpretation(
                forecast_results.get('forecasts'),
                forecast_results.get('diagnostics')
            )
            if forecast_interp:
                report.append("\n## 4. Forecast Results")
                report.append(forecast_interp)
        
        # Conclusions
        report.append("\n## 5. Conclusions and Recommendations")
        
        conclusions = []
        
        # Data quality conclusions
        if data is not None:
            missing_pct = data.isnull().sum().sum() / len(data) * 100
            if missing_pct < 5:
                conclusions.append("✅ Data quality is good with minimal missing values")
            else:
                conclusions.append("⚠️ Data quality issues detected - consider additional preprocessing")
        
        # Model conclusions
        if model_results and model_results.get('validation', {}).get('overall', {}).get('is_valid', False):
            conclusions.append("✅ SARIMA model is suitable for this time series")
        else:
            conclusions.append("⚠️ Model may need adjustment for better performance")
        
        # Forecast conclusions
        if forecast_results:
            conclusions.append("📊 Forecasts generated with confidence intervals")
            conclusions.append("🔄 Regular model updates recommended as new data becomes available")
        
        if conclusions:
            report.extend(conclusions)
        
        report.append("\n" + "="*60)
        report.append("End of Report")
        
        return "\n".join(report), None
        
    except Exception as e:
        return None, f"Error generating comprehensive report: {str(e)}"

def interpret_acf_pacf(acf_values, pacf_values):
    """Interpret ACF and PACF plots for model selection"""
    try:
        interpretation = []
        
        interpretation.append("**ACF and PACF Interpretation:**\n")
        
        # Analyze ACF pattern
        significant_acf = np.where(np.abs(acf_values[1:]) > 0.2)[0] + 1
        
        if len(significant_acf) == 0:
            interpretation.append("- ACF: No significant autocorrelations detected")
        else:
            interpretation.append(f"- ACF: Significant autocorrelations at lags {significant_acf[:5].tolist()}")
        
        # Analyze PACF pattern
        significant_pacf = np.where(np.abs(pacf_values[1:]) > 0.2)[0] + 1
        
        if len(significant_pacf) == 0:
            interpretation.append("- PACF: No significant partial autocorrelations detected")
        else:
            interpretation.append(f"- PACF: Significant partial autocorrelations at lags {significant_pacf[:5].tolist()}")
        
        # Model suggestions
        interpretation.append(f"\n**Model Suggestions:**")
        
        if len(significant_acf) > 0 and len(significant_pacf) > 0:
            interpretation.append("- Consider ARMA model with both AR and MA components")
            interpretation.append(f"- Suggested AR order: {min(significant_pacf[0], 3)}")
            interpretation.append(f"- Suggested MA order: {min(significant_acf[0], 3)}")
        elif len(significant_pacf) > 0:
            interpretation.append("- Consider AR model")
            interpretation.append(f"- Suggested AR order: {min(significant_pacf[0], 3)}")
        elif len(significant_acf) > 0:
            interpretation.append("- Consider MA model")
            interpretation.append(f"- Suggested MA order: {min(significant_acf[0], 3)}")
        else:
            interpretation.append("- Data may be white noise or require differencing")
        
        return "\n".join(interpretation), None
        
    except Exception as e:
        return None, f"Error interpreting ACF/PACF: {str(e)}"

def generate_executive_summary(analysis_results):
    """Generate an executive summary of the analysis"""
    try:
        summary = []
        
        summary.append("**Executive Summary:**\n")
        
        # Key findings
        key_findings = []
        
        if analysis_results.get('data_quality', {}).get('missing_percentage', 0) < 5:
            key_findings.append("High data quality with minimal missing values")
        
        if analysis_results.get('patterns', {}).get('trend', {}).get('present', False):
            trend_dir = analysis_results['patterns']['trend']['direction']
            key_findings.append(f"Clear {trend_dir} trend identified")
        
        if analysis_results.get('patterns', {}).get('seasonality', {}).get('present', False):
            key_findings.append("Seasonal patterns detected")
        
        if analysis_results.get('model_validation', {}).get('overall', {}).get('is_valid', False):
            key_findings.append("SARIMA model validation successful")
        
        if key_findings:
            summary.append("**Key Findings:**")
            for finding in key_findings:
                summary.append(f"• {finding}")
        
        # Recommendations
        recommendations = [
            "Monitor forecasts against actual values",
            "Update model monthly with new data",
            "Consider external factors that may affect the series",
            "Implement automated alerts for forecast deviations"
        ]
        
        summary.append(f"\n**Recommendations:**")
        for rec in recommendations:
            summary.append(f"• {rec}")
        
        return "\n".join(summary), None
        
    except Exception as e:
        return None, f"Error generating executive summary: {str(e)}"
