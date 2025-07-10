import streamlit as st
import pandas as pd
import numpy as np
from utils.forecasting import (
    generate_forecasts, calculate_forecast_metrics, backtest_model,
    cross_validate_model, forecast_scenarios, export_forecasts,
    calculate_forecast_intervals, forecast_diagnostics
)
from utils.visualization import create_forecast_plot, create_time_series_plot
from utils.interpretation import (
    generate_forecast_interpretation, generate_comprehensive_report
)
from utils.data_import import initialize_session_state

# Initialize session state
initialize_session_state()

st.title("🔮 Forecasting")
st.markdown("Generate forecasts and predictions using your validated SARIMA models.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No data available. Please load data first using the **Data Import** page.")
    st.stop()

data = st.session_state.data

# Check if models are available
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
    st.warning("⚠️ No models available. Please build a model first using the **Modeling** page.")
    st.stop()

# Forecasting overview
st.subheader("📊 Forecasting Overview")

# Create tabs for different forecasting approaches
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Generate Forecasts", 
    "📊 Model Validation", 
    "🎯 Forecast Scenarios",
    "📈 Forecast Analysis",
    "📋 Final Report"
])

with tab1:
    st.subheader("Generate Forecasts")
    
    # Model selection
    selected_model_name = st.selectbox("Select model for forecasting:", list(available_models.keys()))
    selected_model = available_models[selected_model_name]
    
    # Display model information
    st.info(f"**Model:** SARIMA{selected_model['order']}×{selected_model['seasonal_order']}")
    
    # Forecast parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_steps = st.number_input(
            "Forecast Horizon (periods)",
            min_value=1,
            max_value=365,
            value=12,
            help="Number of periods to forecast into the future"
        )
    
    with col2:
        confidence_level = st.selectbox(
            "Confidence Level",
            [0.80, 0.90, 0.95, 0.99],
            index=2,
            help="Confidence level for forecast intervals"
        )
    
    # Generate forecasts
    if st.button("Generate Forecasts"):
        with st.spinner("Generating forecasts..."):
            forecast_result, error = generate_forecasts(selected_model, steps=forecast_steps)
            
            if error:
                st.error(f"Error generating forecasts: {error}")
            else:
                st.session_state.forecasts = forecast_result
                st.session_state.selected_model = selected_model
                
                st.success("Forecasts generated successfully!")
                
                # Display forecast results
                st.subheader("📊 Forecast Results")
                
                forecasts_df = forecast_result['forecasts']
                
                # Show forecast table
                st.write("**Forecast Table:**")
                st.dataframe(forecasts_df)
                
                # Create forecast plot
                fig, error = create_forecast_plot(
                    data,
                    forecasts_df,
                    {
                        'lower': forecasts_df['lower_ci'],
                        'upper': forecasts_df['upper_ci']
                    },
                    "Time Series Forecast"
                )
                
                if error:
                    st.error(f"Error creating forecast plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Forecast statistics
                st.subheader("📈 Forecast Statistics")
                
                forecast_values = forecasts_df['forecast']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Forecast", f"{forecast_values.mean():.2f}")
                
                with col2:
                    st.metric("Forecast Std", f"{forecast_values.std():.2f}")
                
                with col3:
                    st.metric("Min Forecast", f"{forecast_values.min():.2f}")
                
                with col4:
                    st.metric("Max Forecast", f"{forecast_values.max():.2f}")
                
                # Forecast trend
                if len(forecast_values) > 1:
                    trend = "Increasing" if forecast_values.iloc[-1] > forecast_values.iloc[0] else "Decreasing"
                    st.info(f"**Forecast Trend:** {trend}")
                
                # Export forecasts
                st.subheader("📥 Export Forecasts")
                
                if st.button("Export Forecast Data"):
                    csv_data, error = export_forecasts(forecast_result)
                    
                    if error:
                        st.error(f"Error exporting forecasts: {error}")
                    else:
                        st.download_button(
                            label="Download Forecasts as CSV",
                            data=csv_data,
                            file_name=f"forecasts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("Forecast data exported successfully!")

with tab2:
    st.subheader("Model Validation")
    
    st.markdown("""
    **Model Validation** tests the forecasting performance of your model using historical data.
    """)
    
    # Validation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test Size (proportion)",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
    
    with col2:
        selected_model_name = st.selectbox(
            "Select model to validate:",
            list(available_models.keys()),
            key="validation_model"
        )
    
    # Backtesting
    st.subheader("📊 Backtesting")
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            selected_model = available_models[selected_model_name]
            
            backtest_result, error = backtest_model(
                data,
                selected_model['order'],
                selected_model['seasonal_order'],
                test_size
            )
            
            if error:
                st.error(f"Error in backtesting: {error}")
            else:
                st.session_state.backtest_result = backtest_result
                
                st.success("Backtesting completed!")
                
                # Display backtest results
                metrics = backtest_result['metrics']
                
                st.write("**Forecast Accuracy Metrics:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                
                with col2:
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    st.metric("MASE", f"{metrics['mase']:.2f}")
                
                with col3:
                    st.metric("R²", f"{metrics['r_squared']:.3f}")
                    st.metric("Theil's U", f"{metrics['theil_u']:.3f}")
                
                # Interpretation of metrics
                st.subheader("📝 Metrics Interpretation")
                
                interpretations = []
                
                if metrics['mape'] < 10:
                    interpretations.append("✅ Excellent forecast accuracy (MAPE < 10%)")
                elif metrics['mape'] < 20:
                    interpretations.append("✅ Good forecast accuracy (MAPE < 20%)")
                elif metrics['mape'] < 50:
                    interpretations.append("⚠️ Moderate forecast accuracy (MAPE < 50%)")
                else:
                    interpretations.append("❌ Poor forecast accuracy (MAPE > 50%)")
                
                if metrics['mase'] < 1:
                    interpretations.append("✅ Better than naive forecast (MASE < 1)")
                else:
                    interpretations.append("⚠️ Worse than naive forecast (MASE > 1)")
                
                if metrics['r_squared'] > 0.7:
                    interpretations.append("✅ Strong predictive power (R² > 0.7)")
                elif metrics['r_squared'] > 0.5:
                    interpretations.append("✅ Moderate predictive power (R² > 0.5)")
                else:
                    interpretations.append("⚠️ Weak predictive power (R² < 0.5)")
                
                for interp in interpretations:
                    st.write(interp)
                
                # Backtest plot
                fig, error = create_forecast_plot(
                    backtest_result['train_data'],
                    backtest_result['forecasts'],
                    title="Backtesting Results"
                )
                
                if error:
                    st.error(f"Error creating backtest plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation
    st.subheader("🔄 Cross-Validation")
    
    n_splits = st.slider(
        "Number of CV Splits",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of cross-validation splits"
    )
    
    if st.button("Run Cross-Validation"):
        with st.spinner("Running cross-validation..."):
            selected_model = available_models[selected_model_name]
            
            cv_result, error = cross_validate_model(
                data,
                selected_model['order'],
                selected_model['seasonal_order'],
                n_splits
            )
            
            if error:
                st.error(f"Error in cross-validation: {error}")
            else:
                st.session_state.cv_result = cv_result
                
                st.success("Cross-validation completed!")
                
                # Display CV results
                avg_metrics = cv_result['average_metrics']
                
                st.write("**Cross-Validation Results:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg RMSE", f"{avg_metrics['rmse']:.2f}")
                    st.metric("RMSE Std", f"{avg_metrics['rmse_std']:.2f}")
                
                with col2:
                    st.metric("Avg MAE", f"{avg_metrics['mae']:.2f}")
                    st.metric("MAE Std", f"{avg_metrics['mae_std']:.2f}")
                
                with col3:
                    st.metric("Avg MAPE", f"{avg_metrics['mape']:.2f}%")
                    st.metric("MAPE Std", f"{avg_metrics['mape_std']:.2f}%")
                
                # CV interpretation
                st.write("**Cross-Validation Interpretation:**")
                
                if avg_metrics['mape'] < 15:
                    st.success("✅ Consistent good performance across folds")
                elif avg_metrics['mape'] < 30:
                    st.info("✅ Moderate performance with some variability")
                else:
                    st.warning("⚠️ High variability in performance across folds")

with tab3:
    st.subheader("Forecast Scenarios")
    
    st.markdown("""
    **Forecast Scenarios** generate different prediction outcomes based on confidence intervals.
    """)
    
    # Check if forecasts are available
    if 'forecasts' not in st.session_state:
        st.info("Please generate forecasts first in the **Generate Forecasts** tab.")
    else:
        # Scenario generation
        if st.button("Generate Forecast Scenarios"):
            with st.spinner("Generating forecast scenarios..."):
                selected_model = st.session_state.selected_model
                
                scenarios_result, error = forecast_scenarios(
                    selected_model,
                    steps=st.session_state.forecasts['steps'],
                    scenarios=['optimistic', 'pessimistic', 'realistic']
                )
                
                if error:
                    st.error(f"Error generating scenarios: {error}")
                else:
                    st.session_state.scenarios = scenarios_result
                    
                    st.success("Forecast scenarios generated!")
                    
                    # Display scenarios
                    st.subheader("📊 Forecast Scenarios")
                    
                    scenarios_df = scenarios_result['scenarios']
                    
                    # Show scenarios table
                    st.write("**Scenario Table:**")
                    st.dataframe(scenarios_df)
                    
                    # Create scenarios plot
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data.iloc[:, 0],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add scenarios
                    colors = {'realistic': 'green', 'optimistic': 'orange', 'pessimistic': 'red'}
                    
                    for scenario in scenarios_df.columns:
                        fig.add_trace(go.Scatter(
                            x=scenarios_df.index,
                            y=scenarios_df[scenario],
                            mode='lines',
                            name=scenario.capitalize(),
                            line=dict(color=colors.get(scenario, 'gray'), width=2, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title="Forecast Scenarios",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode='x unified',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Scenario analysis
                    st.subheader("📈 Scenario Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Optimistic Scenario:**")
                        opt_values = scenarios_df['optimistic']
                        st.metric("Mean", f"{opt_values.mean():.2f}")
                        st.metric("Max", f"{opt_values.max():.2f}")
                    
                    with col2:
                        st.write("**Realistic Scenario:**")
                        real_values = scenarios_df['realistic']
                        st.metric("Mean", f"{real_values.mean():.2f}")
                        st.metric("Range", f"{real_values.max() - real_values.min():.2f}")
                    
                    with col3:
                        st.write("**Pessimistic Scenario:**")
                        pess_values = scenarios_df['pessimistic']
                        st.metric("Mean", f"{pess_values.mean():.2f}")
                        st.metric("Min", f"{pess_values.min():.2f}")
        
        # Multiple confidence intervals
        st.subheader("🎯 Multiple Confidence Intervals")
        
        confidence_levels = st.multiselect(
            "Select confidence levels:",
            [0.80, 0.90, 0.95, 0.99],
            default=[0.90, 0.95]
        )
        
        if confidence_levels and st.button("Generate Confidence Intervals"):
            with st.spinner("Calculating confidence intervals..."):
                selected_model = st.session_state.selected_model
                
                intervals_result, error = calculate_forecast_intervals(
                    selected_model,
                    steps=st.session_state.forecasts['steps'],
                    confidence_levels=confidence_levels
                )
                
                if error:
                    st.error(f"Error calculating intervals: {error}")
                else:
                    st.session_state.intervals = intervals_result
                    
                    st.success("Confidence intervals calculated!")
                    
                    # Display intervals
                    st.write("**Confidence Intervals:**")
                    
                    point_forecasts = intervals_result['point_forecasts']
                    
                    # Create intervals plot
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data.iloc[:, 0],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add point forecasts
                    fig.add_trace(go.Scatter(
                        x=intervals_result['forecast_dates'],
                        y=point_forecasts,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add confidence intervals
                    colors = ['rgba(255,0,0,0.1)', 'rgba(255,0,0,0.2)', 'rgba(255,0,0,0.3)', 'rgba(255,0,0,0.4)']
                    
                    for i, level in enumerate(confidence_levels):
                        level_key = f'{int(level*100)}%'
                        intervals = intervals_result['confidence_intervals'][level_key]
                        
                        fig.add_trace(go.Scatter(
                            x=list(intervals_result['forecast_dates']) + list(intervals_result['forecast_dates'][::-1]),
                            y=list(intervals['upper']) + list(intervals['lower'][::-1]),
                            fill='toself',
                            fillcolor=colors[i % len(colors)],
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{level_key} CI'
                        ))
                    
                    fig.update_layout(
                        title="Forecasts with Multiple Confidence Intervals",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode='x unified',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Forecast Analysis")
    
    st.markdown("""
    **Forecast Analysis** provides detailed analysis and interpretation of your forecasts.
    """)
    
    # Check if forecasts are available
    if 'forecasts' not in st.session_state:
        st.info("Please generate forecasts first in the **Generate Forecasts** tab.")
    else:
        forecasts = st.session_state.forecasts
        
        # Forecast diagnostics
        if st.button("Run Forecast Diagnostics"):
            with st.spinner("Running forecast diagnostics..."):
                diagnostics, error = forecast_diagnostics(forecasts)
                
                if error:
                    st.error(f"Error in forecast diagnostics: {error}")
                else:
                    st.session_state.forecast_diagnostics = diagnostics
                    
                    st.success("Forecast diagnostics completed!")
                    
                    # Display diagnostics
                    st.subheader("📊 Forecast Diagnostics")
                    
                    # Forecast statistics
                    if 'forecast_stats' in diagnostics:
                        stats = diagnostics['forecast_stats']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mean Forecast", f"{stats['mean']:.2f}")
                            st.metric("Forecast Std", f"{stats['std']:.2f}")
                        
                        with col2:
                            st.metric("Min Forecast", f"{stats['min']:.2f}")
                            st.metric("Max Forecast", f"{stats['max']:.2f}")
                        
                        with col3:
                            st.metric("Trend", stats['trend'])
                    
                    # Uncertainty analysis
                    if 'uncertainty' in diagnostics:
                        uncertainty = diagnostics['uncertainty']
                        
                        st.subheader("📈 Uncertainty Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Avg CI Width", f"{uncertainty['avg_ci_width']:.2f}")
                            st.metric("Max CI Width", f"{uncertainty['max_ci_width']:.2f}")
                        
                        with col2:
                            st.metric("Min CI Width", f"{uncertainty['min_ci_width']:.2f}")
                            
                            if uncertainty['increasing_uncertainty']:
                                st.warning("⚠️ Uncertainty increases over time")
                            else:
                                st.success("✅ Stable uncertainty")
                    
                    # Forecast interpretation
                    interpretation, error = generate_forecast_interpretation(
                        forecasts,
                        diagnostics
                    )
                    
                    if error:
                        st.error(f"Error generating interpretation: {error}")
                    else:
                        st.subheader("📝 Forecast Interpretation")
                        st.markdown(interpretation)
        
        # Forecast comparison with actual (if available)
        st.subheader("📊 Forecast vs Actual Comparison")
        
        st.info("This feature would compare forecasts with actual values as they become available.")
        
        # Forecast export options
        st.subheader("📥 Export Forecast Analysis")
        
        if st.button("Export Forecast Report"):
            with st.spinner("Generating forecast report..."):
                # Create comprehensive forecast report
                report = []
                
                report.append("# Forecast Analysis Report")
                report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report.append("="*60)
                
                # Forecast summary
                report.append("\n## Forecast Summary")
                report.append(f"- Forecast horizon: {forecasts['steps']} periods")
                report.append(f"- Model: SARIMA{st.session_state.selected_model['order']}×{st.session_state.selected_model['seasonal_order']}")
                
                # Forecast statistics
                if 'forecast_diagnostics' in st.session_state:
                    diagnostics = st.session_state.forecast_diagnostics
                    
                    if 'forecast_stats' in diagnostics:
                        stats = diagnostics['forecast_stats']
                        report.append(f"\n## Forecast Statistics")
                        report.append(f"- Mean forecast: {stats['mean']:.2f}")
                        report.append(f"- Forecast standard deviation: {stats['std']:.2f}")
                        report.append(f"- Forecast range: {stats['min']:.2f} to {stats['max']:.2f}")
                        report.append(f"- Trend: {stats['trend']}")
                
                # Validation results
                if 'backtest_result' in st.session_state:
                    backtest = st.session_state.backtest_result
                    metrics = backtest['metrics']
                    
                    report.append(f"\n## Validation Results")
                    report.append(f"- RMSE: {metrics['rmse']:.2f}")
                    report.append(f"- MAE: {metrics['mae']:.2f}")
                    report.append(f"- MAPE: {metrics['mape']:.2f}%")
                    report.append(f"- R-squared: {metrics['r_squared']:.3f}")
                
                # Forecast interpretation
                if 'forecast_diagnostics' in st.session_state:
                    interpretation, _ = generate_forecast_interpretation(
                        forecasts,
                        st.session_state.forecast_diagnostics
                    )
                    
                    if interpretation:
                        report.append(f"\n## Forecast Interpretation")
                        report.append(interpretation)
                
                report_text = "\n".join(report)
                
                st.download_button(
                    label="Download Forecast Report",
                    data=report_text,
                    file_name=f"forecast_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                st.success("Forecast report generated!")

with tab5:
    st.subheader("Comprehensive Final Report")
    
    st.markdown("""
    **Final Report** generates a comprehensive analysis report including data overview, 
    analysis results, model performance, and forecasting results.
    """)
    
    if st.button("Generate Final Report"):
        with st.spinner("Generating comprehensive final report..."):
            # Collect all analysis results
            analysis_results = {
                'decomposition': st.session_state.get('decomposition'),
                'patterns': st.session_state.get('patterns'),
                'recommendations': st.session_state.get('recommendations')
            }
            
            model_results = {
                'model': st.session_state.get('selected_model'),
                'validation': st.session_state.get('validation_result')
            }
            
            forecast_results = {
                'forecasts': st.session_state.get('forecasts'),
                'diagnostics': st.session_state.get('forecast_diagnostics'),
                'backtest': st.session_state.get('backtest_result')
            }
            
            # Generate comprehensive report
            report, error = generate_comprehensive_report(
                data,
                analysis_results,
                model_results,
                forecast_results
            )
            
            if error:
                st.error(f"Error generating report: {error}")
            else:
                st.success("Final report generated!")
                
                # Display report
                st.subheader("📋 Comprehensive Analysis Report")
                st.markdown(report)
                
                # Download report
                st.download_button(
                    label="📥 Download Final Report",
                    data=report,
                    file_name=f"time_flow_final_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

# Forecasting status
st.markdown("---")
st.subheader("📋 Forecasting Status")

forecasting_completed = []

if 'forecasts' in st.session_state:
    forecasting_completed.append("✅ Forecasts Generated")

if 'backtest_result' in st.session_state:
    forecasting_completed.append("✅ Model Backtesting")

if 'cv_result' in st.session_state:
    forecasting_completed.append("✅ Cross-Validation")

if 'scenarios' in st.session_state:
    forecasting_completed.append("✅ Forecast Scenarios")

if 'forecast_diagnostics' in st.session_state:
    forecasting_completed.append("✅ Forecast Analysis")

if forecasting_completed:
    st.success("**Forecasting Complete:**")
    for item in forecasting_completed:
        st.write(item)
    
    # Key insights
    st.subheader("🎯 Key Insights")
    
    insights = []
    
    if 'forecasts' in st.session_state:
        forecasts = st.session_state.forecasts
        forecast_values = forecasts['forecasts']['forecast']
        
        if len(forecast_values) > 1:
            if forecast_values.iloc[-1] > forecast_values.iloc[0]:
                insights.append("📈 Forecasts show an upward trend")
            else:
                insights.append("📉 Forecasts show a downward trend")
    
    if 'backtest_result' in st.session_state:
        backtest = st.session_state.backtest_result
        mape = backtest['metrics']['mape']
        
        if mape < 10:
            insights.append("✅ Excellent forecast accuracy achieved")
        elif mape < 20:
            insights.append("✅ Good forecast accuracy achieved")
        else:
            insights.append("⚠️ Consider model improvements for better accuracy")
    
    if not insights:
        insights.append("Complete forecasting steps above to see insights")
    
    for insight in insights:
        st.write(insight)
    
    # Export all results
    st.subheader("📥 Export All Results")
    
    if st.button("Export All Forecasting Results"):
        # Create comprehensive export
        export_data = {
            'forecasts': st.session_state.get('forecasts', {}).get('forecasts', pd.DataFrame()),
            'backtest_metrics': st.session_state.get('backtest_result', {}).get('metrics', {}),
            'cv_results': st.session_state.get('cv_result', {}).get('average_metrics', {}),
            'scenarios': st.session_state.get('scenarios', {}).get('scenarios', pd.DataFrame())
        }
        
        # Create Excel file with multiple sheets
        try:
            import io
            
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Forecasts sheet
                if not export_data['forecasts'].empty:
                    export_data['forecasts'].to_excel(writer, sheet_name='Forecasts')
                
                # Scenarios sheet
                if not export_data['scenarios'].empty:
                    export_data['scenarios'].to_excel(writer, sheet_name='Scenarios')
                
                # Metrics sheet
                metrics_df = pd.DataFrame({
                    'Metric': list(export_data['backtest_metrics'].keys()),
                    'Value': list(export_data['backtest_metrics'].values())
                })
                
                if not metrics_df.empty:
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            buffer.seek(0)
            
            st.download_button(
                label="Download All Results (Excel)",
                data=buffer,
                file_name=f"forecasting_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("All forecasting results exported!")
            
        except ImportError:
            st.warning("Excel export requires xlsxwriter. Using CSV export instead.")
            
            # Fallback to CSV export
            if not export_data['forecasts'].empty:
                csv_data = export_data['forecasts'].to_csv()
                st.download_button(
                    label="Download Forecasts (CSV)",
                    data=csv_data,
                    file_name=f"forecasting_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

else:
    st.info("No forecasting completed yet. Use the tabs above to generate and analyze forecasts.")
    
    # Quick start
    st.subheader("🚀 Quick Start")
    
    if st.button("Quick Forecast (12 periods)"):
        if available_models:
            # Use first available model
            model_name = list(available_models.keys())[0]
            model = available_models[model_name]
            
            with st.spinner("Generating quick forecast..."):
                forecast_result, error = generate_forecasts(model, steps=12)
                
                if not error:
                    st.session_state.forecasts = forecast_result
                    st.session_state.selected_model = model
                    st.success("Quick forecast generated! See results above.")
                    st.rerun()
                else:
                    st.error(f"Error generating quick forecast: {error}")
