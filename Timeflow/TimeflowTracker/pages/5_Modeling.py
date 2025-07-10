import streamlit as st
import pandas as pd
import numpy as np
from utils.sarima_modeling import (
    fit_sarima_model, validate_sarima_model, auto_arima_selection,
    diagnose_model_fit, compare_models, extract_model_components,
    generate_model_report
)
from utils.interpretation import generate_model_interpretation
from utils.data_import import initialize_session_state

# Initialize session state
initialize_session_state()

st.title("🎯 SARIMA Modeling")
st.markdown("Build and validate SARIMA models for your time series data.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No data available. Please load data first using the **Data Import** page.")
    st.stop()

data = st.session_state.data

# Modeling overview
st.subheader("📊 SARIMA Modeling Overview")

st.markdown("""
**SARIMA** (Seasonal AutoRegressive Integrated Moving Average) models are powerful tools for time series forecasting.

**Parameters:**
- **p**: Number of autoregressive terms
- **d**: Number of differences needed for stationarity
- **q**: Number of moving average terms
- **P**: Number of seasonal autoregressive terms
- **D**: Number of seasonal differences
- **Q**: Number of seasonal moving average terms
- **s**: Seasonal period
""")

# Create tabs for different modeling approaches
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Manual Model", 
    "🤖 Auto Model Selection", 
    "✅ Model Validation",
    "📊 Model Comparison",
    "📋 Model Report"
])

with tab1:
    st.subheader("Manual SARIMA Model Configuration")
    
    # Model parameter configuration
    st.write("**Non-seasonal Parameters (p, d, q)**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
    
    with col2:
        d = st.number_input("d (Differencing)", min_value=0, max_value=3, value=1)
    
    with col3:
        q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
    
    st.write("**Seasonal Parameters (P, D, Q, s)**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        P = st.number_input("P (Seasonal AR)", min_value=0, max_value=3, value=1)
    
    with col2:
        D = st.number_input("D (Seasonal Diff)", min_value=0, max_value=2, value=1)
    
    with col3:
        Q = st.number_input("Q (Seasonal MA)", min_value=0, max_value=3, value=1)
    
    with col4:
        s = st.number_input("s (Seasonal Period)", min_value=2, max_value=365, value=12)
    
    # Display model specification
    st.info(f"**Model Specification:** SARIMA({p},{d},{q})×({P},{D},{Q},{s})")
    
    # Fit model
    if st.button("Fit SARIMA Model"):
        with st.spinner("Fitting SARIMA model..."):
            model_result, error = fit_sarima_model(
                data, 
                order=(p, d, q),
                seasonal_order=(P, D, Q, s)
            )
            
            if error:
                st.error(f"Error fitting model: {error}")
            else:
                st.session_state.manual_model = model_result
                
                st.success("Model fitted successfully!")
                
                # Display model summary
                st.subheader("📊 Model Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("AIC", f"{model_result['aic']:.2f}")
                    st.metric("BIC", f"{model_result['bic']:.2f}")
                
                with col2:
                    st.metric("Log-Likelihood", f"{model_result['model'].llf:.2f}")
                    st.metric("Observations", len(data))
                
                # Model diagnostics
                st.subheader("🔍 Model Diagnostics")
                
                diagnostics, error = diagnose_model_fit(model_result)
                
                if error:
                    st.error(f"Error in diagnostics: {error}")
                else:
                    # Display diagnostics
                    if 'fit_metrics' in diagnostics:
                        st.write("**Fit Metrics:**")
                        fit_metrics = diagnostics['fit_metrics']
                        st.write(f"- AIC: {fit_metrics['aic']:.2f}")
                        st.write(f"- BIC: {fit_metrics['bic']:.2f}")
                    
                    if 'parameters' in diagnostics:
                        params = diagnostics['parameters']
                        if 'significance_ratio' in params:
                            st.write(f"**Parameter Significance:** {params['significance_ratio']:.1%}")
                    
                    if 'recommendations' in diagnostics:
                        st.write("**Recommendations:**")
                        for rec in diagnostics['recommendations']:
                            st.write(f"- {rec}")
                
                # Model parameters
                st.subheader("📋 Model Parameters")
                
                try:
                    params_df = pd.DataFrame({
                        'Parameter': model_result['model'].params.index,
                        'Coefficient': model_result['model'].params.values,
                        'Std Error': model_result['model'].bse.values,
                        'P-value': model_result['model'].pvalues.values
                    })
                    
                    st.dataframe(params_df)
                    
                except Exception as e:
                    st.warning("Could not display parameter table")
                
                # Residual analysis
                st.subheader("📊 Residual Analysis")
                
                residuals = model_result['residuals']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Residual Mean", f"{residuals.mean():.4f}")
                
                with col2:
                    st.metric("Residual Std", f"{residuals.std():.4f}")
                
                with col3:
                    st.metric("Residual Min/Max", f"{residuals.min():.2f} / {residuals.max():.2f}")
                
                # Residual plot
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Add residual time series
                fig.add_trace(go.Scatter(
                    x=residuals.index,
                    y=residuals.values,
                    mode='lines+markers',
                    name='Residuals',
                    line=dict(color='red', width=1),
                    marker=dict(size=3)
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                
                fig.update_layout(
                    title="Model Residuals",
                    xaxis_title="Date",
                    yaxis_title="Residual",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Automatic Model Selection")
    
    st.markdown("""
    **Auto ARIMA** automatically selects the best SARIMA parameters by testing multiple combinations
    and choosing the model with the lowest AIC.
    """)
    
    # Auto ARIMA parameters
    st.write("**Search Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Non-seasonal limits:**")
        max_p = st.number_input("Max p", min_value=0, max_value=5, value=3)
        max_d = st.number_input("Max d", min_value=0, max_value=3, value=2)
        max_q = st.number_input("Max q", min_value=0, max_value=5, value=3)
    
    with col2:
        st.write("**Seasonal limits:**")
        max_P = st.number_input("Max P", min_value=0, max_value=3, value=2)
        max_D = st.number_input("Max D", min_value=0, max_value=2, value=1)
        max_Q = st.number_input("Max Q", min_value=0, max_value=3, value=2)
        m = st.number_input("Seasonal period (m)", min_value=2, max_value=365, value=12)
    
    # Estimate computation time
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1) * (max_P + 1) * (max_D + 1) * (max_Q + 1)
    st.info(f"Will test approximately {total_combinations} model combinations")
    
    if st.button("Run Auto ARIMA"):
        with st.spinner("Running automatic model selection... This may take a few minutes."):
            selection_result, error = auto_arima_selection(
                data, 
                max_p=max_p, max_d=max_d, max_q=max_q,
                max_P=max_P, max_D=max_D, max_Q=max_Q,
                m=m
            )
            
            if error:
                st.error(f"Error in auto ARIMA: {error}")
            else:
                st.session_state.auto_model = selection_result
                
                st.success("Auto ARIMA completed!")
                
                # Display best model
                st.subheader("🏆 Best Model")
                
                best_order = selection_result['best_order']
                best_seasonal_order = selection_result['best_seasonal_order']
                best_aic = selection_result['best_aic']
                
                st.info(f"**Best Model:** SARIMA{best_order}×{best_seasonal_order}")
                st.write(f"**Best AIC:** {best_aic:.2f}")
                
                # Display top models
                st.subheader("📊 Top Models")
                
                top_models = selection_result['all_results'][:5]  # Top 5 models
                
                model_comparison = []
                for i, model in enumerate(top_models):
                    model_comparison.append({
                        'Rank': i + 1,
                        'Order': f"{model['order']}×{model['seasonal_order']}",
                        'AIC': f"{model['aic']:.2f}",
                        'BIC': f"{model['bic']:.2f}"
                    })
                
                comparison_df = pd.DataFrame(model_comparison)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Model diagnostics for best model
                st.subheader("🔍 Best Model Diagnostics")
                
                best_model_result = {
                    'model': selection_result['best_model'],
                    'order': best_order,
                    'seasonal_order': best_seasonal_order,
                    'aic': best_aic,
                    'bic': selection_result['best_model'].bic,
                    'fitted_values': selection_result['fitted_values'],
                    'residuals': selection_result['residuals']
                }
                
                diagnostics, error = diagnose_model_fit(best_model_result)
                
                if not error and 'recommendations' in diagnostics:
                    for rec in diagnostics['recommendations']:
                        st.write(f"- {rec}")

with tab3:
    st.subheader("Model Validation")
    
    st.markdown("""
    **Model Validation** tests whether the fitted model adequately captures the underlying patterns
    in the data through residual analysis and diagnostic tests.
    """)
    
    # Check if any model is available
    model_to_validate = None
    
    if 'manual_model' in st.session_state:
        if st.radio("Select model to validate:", ["Manual Model"]) == "Manual Model":
            model_to_validate = st.session_state.manual_model
    
    if 'auto_model' in st.session_state:
        model_choice = st.radio("Select model to validate:", ["Manual Model", "Auto Selected Model"])
        if model_choice == "Auto Selected Model":
            # Create model result from auto selection
            auto_result = st.session_state.auto_model
            model_to_validate = {
                'model': auto_result['best_model'],
                'order': auto_result['best_order'],
                'seasonal_order': auto_result['best_seasonal_order'],
                'aic': auto_result['best_aic'],
                'bic': auto_result['best_model'].bic,
                'fitted_values': auto_result['fitted_values'],
                'residuals': auto_result['residuals']
            }
    
    if model_to_validate is None:
        st.info("Please fit a model first in the **Manual Model** or **Auto Model Selection** tabs.")
    else:
        if st.button("Validate Model"):
            with st.spinner("Validating model..."):
                validation_result, error = validate_sarima_model(model_to_validate)
                
                if error:
                    st.error(f"Error validating model: {error}")
                else:
                    st.session_state.validation_result = validation_result
                    
                    st.success("Model validation completed!")
                    
                    # Display validation results
                    st.subheader("📊 Validation Results")
                    
                    overall = validation_result['overall']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Tests Passed", f"{overall['passed_tests']}/{overall['total_tests']}")
                    
                    with col2:
                        st.metric("Validation Score", f"{overall['validation_score']:.1%}")
                    
                    with col3:
                        if overall['is_valid']:
                            st.success("✅ Model Valid")
                        else:
                            st.error("❌ Model Issues")
                    
                    # Detailed test results
                    st.subheader("🔍 Diagnostic Tests")
                    
                    # Ljung-Box test
                    if 'ljung_box' in validation_result:
                        ljung_box = validation_result['ljung_box']
                        
                        with st.expander("Ljung-Box Test (Autocorrelation)"):
                            st.write("**Test:** Checks for autocorrelation in residuals")
                            st.write("**H0:** No autocorrelation in residuals")
                            
                            if 'p_value' in ljung_box:
                                st.write(f"**P-value:** {ljung_box['p_value']:.4f}")
                                
                                if ljung_box['passed']:
                                    st.success("✅ No significant autocorrelation detected")
                                else:
                                    st.warning("⚠️ Autocorrelation detected in residuals")
                            else:
                                st.error(f"Test failed: {ljung_box.get('error', 'Unknown error')}")
                    
                    # Jarque-Bera test
                    if 'jarque_bera' in validation_result:
                        jarque_bera = validation_result['jarque_bera']
                        
                        with st.expander("Jarque-Bera Test (Normality)"):
                            st.write("**Test:** Checks for normality of residuals")
                            st.write("**H0:** Residuals are normally distributed")
                            
                            if 'p_value' in jarque_bera:
                                st.write(f"**P-value:** {jarque_bera['p_value']:.4f}")
                                
                                if jarque_bera['passed']:
                                    st.success("✅ Residuals appear normally distributed")
                                else:
                                    st.warning("⚠️ Residuals may not be normally distributed")
                            else:
                                st.error(f"Test failed: {jarque_bera.get('error', 'Unknown error')}")
                    
                    # Shapiro-Wilk test
                    if 'shapiro_wilk' in validation_result:
                        shapiro_wilk = validation_result['shapiro_wilk']
                        
                        with st.expander("Shapiro-Wilk Test (Alternative Normality)"):
                            st.write("**Test:** Alternative test for normality of residuals")
                            st.write("**H0:** Residuals are normally distributed")
                            
                            st.write(f"**P-value:** {shapiro_wilk['p_value']:.4f}")
                            
                            if shapiro_wilk['passed']:
                                st.success("✅ Residuals appear normally distributed")
                            else:
                                st.warning("⚠️ Residuals may not be normally distributed")
                    
                    # Residual statistics
                    st.subheader("📊 Residual Statistics")
                    
                    if 'residual_stats' in validation_result:
                        residual_stats = validation_result['residual_stats']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mean", f"{residual_stats['mean']:.4f}")
                            st.metric("Std Dev", f"{residual_stats['std']:.4f}")
                        
                        with col2:
                            st.metric("Skewness", f"{residual_stats['skewness']:.2f}")
                            st.metric("Kurtosis", f"{residual_stats['kurtosis']:.2f}")
                        
                        with col3:
                            st.metric("Min", f"{residual_stats['min']:.2f}")
                            st.metric("Max", f"{residual_stats['max']:.2f}")
                    
                    # Validation interpretation
                    st.subheader("📝 Validation Interpretation")
                    
                    interpretation, error = generate_model_interpretation(
                        model_to_validate, 
                        validation_result
                    )
                    
                    if error:
                        st.error(f"Error generating interpretation: {error}")
                    else:
                        st.markdown(interpretation)

with tab4:
    st.subheader("Model Comparison")
    
    st.markdown("""
    **Model Comparison** allows you to compare multiple SARIMA models to choose the best one.
    """)
    
    # Collect available models
    available_models = {}
    
    if 'manual_model' in st.session_state:
        available_models['Manual Model'] = st.session_state.manual_model
    
    if 'auto_model' in st.session_state:
        auto_result = st.session_state.auto_model
        available_models['Auto Selected Model'] = {
            'model': auto_result['best_model'],
            'order': auto_result['best_order'],
            'seasonal_order': auto_result['best_seasonal_order'],
            'aic': auto_result['best_aic'],
            'bic': auto_result['best_model'].bic,
            'fitted_values': auto_result['fitted_values'],
            'residuals': auto_result['residuals']
        }
    
    if len(available_models) == 0:
        st.info("No models available for comparison. Please fit models first.")
    elif len(available_models) == 1:
        st.info("Only one model available. Fit more models to enable comparison.")
        
        # Show single model details
        model_name = list(available_models.keys())[0]
        model = available_models[model_name]
        
        st.write(f"**Available Model:** {model_name}")
        st.write(f"**Specification:** SARIMA{model['order']}×{model['seasonal_order']}")
        st.write(f"**AIC:** {model['aic']:.2f}")
        st.write(f"**BIC:** {model['bic']:.2f}")
        
    else:
        # Compare multiple models
        if st.button("Compare Models"):
            with st.spinner("Comparing models..."):
                model_results = list(available_models.values())
                
                comparison, error = compare_models(model_results)
                
                if error:
                    st.error(f"Error comparing models: {error}")
                else:
                    st.success("Model comparison completed!")
                    
                    # Display comparison table
                    st.subheader("📊 Model Comparison Table")
                    
                    comparison_data = []
                    model_names = list(available_models.keys())
                    
                    for i, comp in enumerate(comparison):
                        comparison_data.append({
                            'Model': model_names[i] if i < len(model_names) else f"Model {i+1}",
                            'Specification': f"SARIMA{comp['order']}×{comp['seasonal_order']}",
                            'AIC': f"{comp['aic']:.2f}",
                            'BIC': f"{comp['bic']:.2f}",
                            'AIC Rank': comp['aic_rank'],
                            'BIC Rank': comp['bic_rank']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Best model recommendation
                    st.subheader("🏆 Best Model Recommendation")
                    
                    best_aic_model = min(comparison, key=lambda x: x['aic'])
                    best_bic_model = min(comparison, key=lambda x: x['bic'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Best by AIC:**")
                        best_aic_name = model_names[best_aic_model['model_id'] - 1]
                        st.write(f"- {best_aic_name}")
                        st.write(f"- AIC: {best_aic_model['aic']:.2f}")
                    
                    with col2:
                        st.write("**Best by BIC:**")
                        best_bic_name = model_names[best_bic_model['model_id'] - 1]
                        st.write(f"- {best_bic_name}")
                        st.write(f"- BIC: {best_bic_model['bic']:.2f}")
                    
                    # Overall recommendation
                    if best_aic_model['model_id'] == best_bic_model['model_id']:
                        st.success(f"✅ **Recommended Model:** {model_names[best_aic_model['model_id'] - 1]}")
                    else:
                        st.info("📊 **Different models ranked best by AIC and BIC.** Consider both criteria.")
        
        # Individual model details
        st.subheader("🔍 Individual Model Details")
        
        selected_model = st.selectbox("Select model to view details:", list(available_models.keys()))
        
        if selected_model:
            model = available_models[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model:** {selected_model}")
                st.write(f"**Specification:** SARIMA{model['order']}×{model['seasonal_order']}")
                st.write(f"**AIC:** {model['aic']:.2f}")
                st.write(f"**BIC:** {model['bic']:.2f}")
            
            with col2:
                # Model components
                components, error = extract_model_components(model)
                
                if not error and 'statistics' in components:
                    stats = components['statistics']
                    st.write("**Model Statistics:**")
                    st.write(f"- Log-Likelihood: {stats.get('log_likelihood', 'N/A')}")
                    st.write(f"- Scale: {stats.get('scale', 'N/A')}")

with tab5:
    st.subheader("Model Report")
    
    st.markdown("""
    **Model Report** generates a comprehensive report of your SARIMA model including 
    fit statistics, validation results, and interpretation.
    """)
    
    # Select model for report
    available_models = {}
    
    if 'manual_model' in st.session_state:
        available_models['Manual Model'] = st.session_state.manual_model
    
    if 'auto_model' in st.session_state:
        auto_result = st.session_state.auto_model
        available_models['Auto Selected Model'] = {
            'model': auto_result['best_model'],
            'order': auto_result['best_order'],
            'seasonal_order': auto_result['best_seasonal_order'],
            'aic': auto_result['best_aic'],
            'bic': auto_result['best_model'].bic,
            'fitted_values': auto_result['fitted_values'],
            'residuals': auto_result['residuals']
        }
    
    if len(available_models) == 0:
        st.info("No models available for report generation. Please fit a model first.")
    else:
        selected_model = st.selectbox("Select model for report:", list(available_models.keys()))
        
        if st.button("Generate Model Report"):
            with st.spinner("Generating model report..."):
                model = available_models[selected_model]
                
                # Get validation results
                validation_result = st.session_state.get('validation_result', {})
                
                # Generate report
                report, error = generate_model_report(model, validation_result)
                
                if error:
                    st.error(f"Error generating report: {error}")
                else:
                    st.success("Model report generated!")
                    
                    # Display report sections
                    st.subheader("📋 Model Report")
                    
                    # Model specification
                    if 'model_specification' in report:
                        spec = report['model_specification']
                        
                        st.write("**Model Specification:**")
                        st.write(f"- Order: {spec['order']}")
                        st.write(f"- Seasonal Order: {spec['seasonal_order']}")
                        st.write(f"- AIC: {spec['aic']:.2f}")
                        st.write(f"- BIC: {spec['bic']:.2f}")
                    
                    # Validation summary
                    if 'validation_summary' in report:
                        val_summary = report['validation_summary']
                        
                        st.write("**Validation Summary:**")
                        st.write(f"- Tests Passed: {val_summary['passed_tests']}/{val_summary['total_tests']}")
                        st.write(f"- Validation Score: {val_summary['validation_score']:.1%}")
                        
                        if val_summary['is_valid']:
                            st.success("✅ Model is valid")
                        else:
                            st.warning("⚠️ Model may have issues")
                    
                    # Interpretation
                    if 'interpretation' in report:
                        st.write("**Model Interpretation:**")
                        for interp in report['interpretation']:
                            st.write(f"- {interp}")
                    
                    # Export report
                    full_report = []
                    full_report.append(f"# SARIMA Model Report: {selected_model}")
                    full_report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    full_report.append("="*60)
                    
                    # Add all report sections
                    for section, content in report.items():
                        if section != 'interpretation':
                            full_report.append(f"\n## {section.replace('_', ' ').title()}")
                            full_report.append(str(content))
                    
                    if 'interpretation' in report:
                        full_report.append(f"\n## Interpretation")
                        for interp in report['interpretation']:
                            full_report.append(f"- {interp}")
                    
                    report_text = "\n".join(full_report)
                    
                    st.download_button(
                        label="📥 Download Model Report",
                        data=report_text,
                        file_name=f"model_report_{selected_model.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

# Modeling status
st.markdown("---")
st.subheader("📋 Modeling Status")

models_built = []

if 'manual_model' in st.session_state:
    models_built.append("✅ Manual SARIMA Model")

if 'auto_model' in st.session_state:
    models_built.append("✅ Auto Selected Model")

if 'validation_result' in st.session_state:
    models_built.append("✅ Model Validation Completed")

if models_built:
    st.success("**Models Built:**")
    for model in models_built:
        st.write(model)
    
    # Next steps
    st.subheader("🚀 Next Steps")
    
    next_steps = []
    
    if 'validation_result' in st.session_state:
        validation = st.session_state.validation_result
        if validation.get('overall', {}).get('is_valid', False):
            next_steps.append("✅ Model is validated - ready for forecasting")
        else:
            next_steps.append("⚠️ Consider improving model validation before forecasting")
    
    next_steps.append("📊 Visit the **Forecasting** page to generate predictions")
    next_steps.append("📈 Use the **Visualization** page to plot model results")
    
    for step in next_steps:
        st.write(step)

else:
    st.info("No models built yet. Start by fitting a model using the tabs above.")
