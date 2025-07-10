import streamlit as st
import pandas as pd
import numpy as np
from utils.time_series_analysis import (
    decompose_time_series, test_stationarity, calculate_acf_pacf,
    identify_temporal_patterns, recommend_model, difference_series,
    seasonal_difference, detect_changepoints
)
from utils.visualization import create_decomposition_plot, create_correlation_plot
from utils.interpretation import (
    generate_analysis_interpretation, interpret_acf_pacf,
    generate_executive_summary
)
from utils.data_import import initialize_session_state

# Initialize session state
initialize_session_state()

st.title("🔍 Time Series Analysis")
st.markdown("Analyze temporal patterns, stationarity, and identify suitable models for your data.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No data available. Please load data first using the **Data Import** page.")
    st.stop()

data = st.session_state.data

# Analysis overview
st.subheader("📊 Analysis Overview")

# Create tabs for different analysis types
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stationarity Tests", 
    "🔄 Decomposition Analysis", 
    "📊 Pattern Identification",
    "🎯 Model Recommendations",
    "📋 Comprehensive Report"
])

with tab1:
    st.subheader("Stationarity Analysis")
    
    st.markdown("""
    **Stationarity** is a crucial assumption for many time series models. A stationary time series has:
    - Constant mean over time
    - Constant variance over time
    - No systematic patterns or trends
    """)
    
    # Stationarity tests
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Augmented Dickey-Fuller (ADF) Test**")
        st.info("H0: Series has a unit root (non-stationary)")
        
        if st.button("Run ADF Test"):
            with st.spinner("Running ADF test..."):
                adf_result, error = test_stationarity(data, 'adf')
                
                if error:
                    st.error(f"Error in ADF test: {error}")
                else:
                    st.session_state.adf_result = adf_result
                    
                    # Display results
                    st.write("**Results:**")
                    st.write(f"- Test Statistic: {adf_result['test_statistic']:.4f}")
                    st.write(f"- P-value: {adf_result['p_value']:.4f}")
                    st.write(f"- Interpretation: {adf_result['interpretation']}")
                    
                    if adf_result['is_stationary']:
                        st.success("✅ Series appears to be stationary")
                    else:
                        st.warning("⚠️ Series appears to be non-stationary")
                    
                    # Critical values
                    st.write("**Critical Values:**")
                    for key, value in adf_result['critical_values'].items():
                        st.write(f"- {key}: {value:.4f}")
    
    with col2:
        st.write("**KPSS Test**")
        st.info("H0: Series is stationary")
        
        if st.button("Run KPSS Test"):
            with st.spinner("Running KPSS test..."):
                kpss_result, error = test_stationarity(data, 'kpss')
                
                if error:
                    st.error(f"Error in KPSS test: {error}")
                else:
                    st.session_state.kpss_result = kpss_result
                    
                    # Display results
                    st.write("**Results:**")
                    st.write(f"- Test Statistic: {kpss_result['test_statistic']:.4f}")
                    st.write(f"- P-value: {kpss_result['p_value']:.4f}")
                    st.write(f"- Interpretation: {kpss_result['interpretation']}")
                    
                    if kpss_result['is_stationary']:
                        st.success("✅ Series appears to be stationary")
                    else:
                        st.warning("⚠️ Series appears to be non-stationary")
                    
                    # Critical values
                    st.write("**Critical Values:**")
                    for key, value in kpss_result['critical_values'].items():
                        st.write(f"- {key}: {value:.4f}")
    
    # Combined interpretation
    if 'adf_result' in st.session_state and 'kpss_result' in st.session_state:
        st.subheader("📋 Combined Test Interpretation")
        
        adf_stationary = st.session_state.adf_result['is_stationary']
        kpss_stationary = st.session_state.kpss_result['is_stationary']
        
        if adf_stationary and kpss_stationary:
            st.success("✅ Both tests suggest the series is stationary")
        elif not adf_stationary and not kpss_stationary:
            st.warning("⚠️ Both tests suggest the series is non-stationary")
        else:
            st.info("ℹ️ Tests give conflicting results - further investigation needed")
    
    # Differencing options
    st.subheader("🔄 Make Series Stationary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Regular Differencing**")
        diff_order = st.selectbox("Differencing Order", [1, 2, 3], index=0)
        
        if st.button("Apply Differencing"):
            with st.spinner("Applying differencing..."):
                diff_data, error = difference_series(data, diff_order)
                
                if error:
                    st.error(f"Error applying differencing: {error}")
                else:
                    st.session_state.differenced_data = diff_data
                    
                    st.success(f"Applied {diff_order}-order differencing")
                    
                    # Show differenced data statistics
                    st.write("**Differenced Data Statistics:**")
                    st.write(f"- Mean: {diff_data.mean():.4f}")
                    st.write(f"- Std Dev: {diff_data.std():.4f}")
                    st.write(f"- Records: {len(diff_data)}")
                    
                    # Test stationarity of differenced data
                    if st.button("Test Stationarity of Differenced Data"):
                        diff_df = pd.DataFrame(diff_data)
                        adf_diff, _ = test_stationarity(diff_df, 'adf')
                        
                        if adf_diff:
                            st.write(f"ADF p-value: {adf_diff['p_value']:.4f}")
                            if adf_diff['is_stationary']:
                                st.success("✅ Differenced series is stationary")
                            else:
                                st.warning("⚠️ May need higher order differencing")
    
    with col2:
        st.write("**Seasonal Differencing**")
        seasonal_period = st.number_input("Seasonal Period", min_value=2, max_value=365, value=12)
        
        if st.button("Apply Seasonal Differencing"):
            with st.spinner("Applying seasonal differencing..."):
                seasonal_diff_data, error = seasonal_difference(data, seasonal_period)
                
                if error:
                    st.error(f"Error applying seasonal differencing: {error}")
                else:
                    st.session_state.seasonal_diff_data = seasonal_diff_data
                    
                    st.success(f"Applied seasonal differencing (period={seasonal_period})")
                    
                    # Show seasonal differenced data statistics
                    st.write("**Seasonal Differenced Data Statistics:**")
                    st.write(f"- Mean: {seasonal_diff_data.mean():.4f}")
                    st.write(f"- Std Dev: {seasonal_diff_data.std():.4f}")
                    st.write(f"- Records: {len(seasonal_diff_data)}")

with tab2:
    st.subheader("Time Series Decomposition")
    
    st.markdown("""
    **Decomposition** breaks down a time series into its constituent components:
    - **Trend**: Long-term movement in the data
    - **Seasonality**: Regular patterns that repeat over time
    - **Residual**: Random fluctuations after removing trend and seasonality
    """)
    
    # Decomposition parameters
    col1, col2 = st.columns(2)
    
    with col1:
        decomp_model = st.selectbox(
            "Decomposition Model",
            ["additive", "multiplicative"],
            help="Additive: components are added together. Multiplicative: components are multiplied."
        )
    
    with col2:
        # Auto-suggest period based on data frequency
        suggested_period = 12 if len(data) > 24 else 7 if len(data) > 14 else 4
        period = st.number_input(
            "Seasonal Period",
            min_value=2,
            max_value=min(len(data)//2, 365),
            value=suggested_period,
            help="Number of observations in one seasonal cycle"
        )
    
    if st.button("Perform Decomposition"):
        with st.spinner("Performing decomposition..."):
            decomposition, error = decompose_time_series(data, model=decomp_model, period=period)
            
            if error:
                st.error(f"Error in decomposition: {error}")
            else:
                st.session_state.decomposition = decomposition
                
                st.success("Decomposition completed successfully!")
                
                # Create decomposition plot
                fig, error = create_decomposition_plot(
                    decomposition['trend'],
                    decomposition['seasonal'],
                    decomposition['residual'],
                    f"Time Series Decomposition ({decomp_model})"
                )
                
                if error:
                    st.error(f"Error creating plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Component analysis
                st.subheader("📊 Component Analysis")
                
                # Trend analysis
                trend_component = decomposition['trend'].dropna()
                if len(trend_component) > 1:
                    trend_slope = np.polyfit(range(len(trend_component)), trend_component, 1)[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Trend Direction", 
                                "Increasing" if trend_slope > 0 else "Decreasing")
                        st.metric("Trend Slope", f"{trend_slope:.4f}")
                    
                    with col2:
                        seasonal_component = decomposition['seasonal'].dropna()
                        if len(seasonal_component) > 0:
                            seasonal_strength = seasonal_component.std()
                            st.metric("Seasonal Strength", f"{seasonal_strength:.4f}")
                            st.metric("Seasonal Range", 
                                    f"{seasonal_component.max() - seasonal_component.min():.4f}")
                    
                    with col3:
                        residual_component = decomposition['residual'].dropna()
                        if len(residual_component) > 0:
                            st.metric("Residual Std", f"{residual_component.std():.4f}")
                            st.metric("Residual Mean", f"{residual_component.mean():.4f}")
                
                # Decomposition interpretation
                st.subheader("📝 Decomposition Interpretation")
                
                interpretation = []
                
                if abs(trend_slope) > 0.01:
                    direction = "increasing" if trend_slope > 0 else "decreasing"
                    interpretation.append(f"**Trend**: Clear {direction} trend detected")
                else:
                    interpretation.append("**Trend**: No significant trend detected")
                
                if seasonal_component.std() > 0.1:
                    interpretation.append(f"**Seasonality**: Strong seasonal pattern with period {period}")
                else:
                    interpretation.append("**Seasonality**: Weak or no seasonal pattern")
                
                if residual_component.std() < trend_component.std():
                    interpretation.append("**Residuals**: Low noise level - good decomposition")
                else:
                    interpretation.append("**Residuals**: High noise level - consider different parameters")
                
                for interp in interpretation:
                    st.write(interp)

with tab3:
    st.subheader("Pattern Identification")
    
    st.markdown("""
    **Pattern Identification** analyzes your time series to identify:
    - Stationarity characteristics
    - Trend patterns
    - Seasonal patterns
    - Autocorrelation structure
    """)
    
    if st.button("Identify Temporal Patterns"):
        with st.spinner("Identifying temporal patterns..."):
            patterns, error = identify_temporal_patterns(data)
            
            if error:
                st.error(f"Error identifying patterns: {error}")
            else:
                st.session_state.patterns = patterns
                
                st.success("Pattern identification completed!")
                
                # Display pattern results
                st.subheader("📊 Pattern Analysis Results")
                
                # Stationarity results
                if 'stationarity' in patterns:
                    st.write("**Stationarity Analysis:**")
                    
                    stationarity = patterns['stationarity']
                    
                    if stationarity.get('likely_stationary', False):
                        st.success("✅ Series appears to be stationary")
                    else:
                        st.warning("⚠️ Series appears to be non-stationary")
                    
                    # Show test results
                    adf_result = stationarity.get('adf_test', {})
                    if adf_result:
                        st.write(f"- ADF Test p-value: {adf_result.get('p_value', 'N/A'):.4f}")
                    
                    kpss_result = stationarity.get('kpss_test', {})
                    if kpss_result:
                        st.write(f"- KPSS Test p-value: {kpss_result.get('p_value', 'N/A'):.4f}")
                
                # Trend analysis
                if 'trend' in patterns:
                    st.write("**Trend Analysis:**")
                    
                    trend = patterns['trend']
                    
                    if trend.get('present', False):
                        direction = trend.get('direction', 'unknown')
                        slope = trend.get('slope', 0)
                        st.write(f"- Trend direction: {direction}")
                        st.write(f"- Trend slope: {slope:.6f}")
                    else:
                        st.write("- No significant trend detected")
                
                # Seasonality analysis
                if 'seasonality' in patterns:
                    st.write("**Seasonality Analysis:**")
                    
                    seasonality = patterns['seasonality']
                    
                    if seasonality.get('present', False):
                        strength = seasonality.get('strength', 0)
                        period = seasonality.get('period', 'unknown')
                        st.write(f"- Seasonal pattern detected")
                        st.write(f"- Seasonal strength: {strength:.4f}")
                        st.write(f"- Seasonal period: {period}")
                    else:
                        st.write("- No significant seasonality detected")
                
                # Autocorrelation analysis
                if 'autocorrelation' in patterns:
                    st.write("**Autocorrelation Analysis:**")
                    
                    autocorr = patterns['autocorrelation']
                    
                    sig_acf = autocorr.get('significant_acf_lags', [])
                    sig_pacf = autocorr.get('significant_pacf_lags', [])
                    
                    if sig_acf:
                        st.write(f"- Significant ACF lags: {sig_acf[:10]}")
                    
                    if sig_pacf:
                        st.write(f"- Significant PACF lags: {sig_pacf[:10]}")
                
                # Basic statistics
                if 'statistics' in patterns:
                    st.write("**Basic Statistics:**")
                    
                    stats = patterns['statistics']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean", f"{stats.get('mean', 0):.2f}")
                        st.metric("Std Dev", f"{stats.get('std', 0):.2f}")
                    
                    with col2:
                        st.metric("Skewness", f"{stats.get('skewness', 0):.2f}")
                        st.metric("Kurtosis", f"{stats.get('kurtosis', 0):.2f}")
                    
                    with col3:
                        jb_stat, jb_pvalue = stats.get('jarque_bera_test', (0, 1))
                        st.metric("JB Test p-value", f"{jb_pvalue:.4f}")
                        
                        if jb_pvalue > 0.05:
                            st.success("✅ Normal distribution")
                        else:
                            st.warning("⚠️ Non-normal distribution")
    
    # Change point detection
    st.subheader("🔍 Change Point Detection")
    
    st.markdown("""
    **Change Point Detection** identifies structural breaks or regime changes in your time series.
    """)
    
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.05,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Lower values are more sensitive to changes"
    )
    
    if st.button("Detect Change Points"):
        with st.spinner("Detecting change points..."):
            changepoints, error = detect_changepoints(data, threshold)
            
            if error:
                st.error(f"Error detecting change points: {error}")
            else:
                st.session_state.changepoints = changepoints
                
                if changepoints:
                    st.warning(f"Found {len(changepoints)} potential change points")
                    
                    # Display change points
                    st.write("**Change Points:**")
                    for i, cp in enumerate(changepoints[:10]):  # Show first 10
                        st.write(f"- {i+1}. {cp}")
                    
                    if len(changepoints) > 10:
                        st.write(f"... and {len(changepoints) - 10} more")
                else:
                    st.success("✅ No significant change points detected")

with tab4:
    st.subheader("Model Recommendations")
    
    st.markdown("""
    **Model Recommendations** suggests appropriate time series models based on the identified patterns.
    """)
    
    # Check if pattern analysis has been performed
    if 'patterns' not in st.session_state:
        st.info("Please perform pattern identification first in the **Pattern Identification** tab.")
        
        if st.button("Quick Pattern Analysis"):
            with st.spinner("Analyzing patterns for recommendations..."):
                patterns, error = identify_temporal_patterns(data)
                
                if not error:
                    st.session_state.patterns = patterns
                    st.rerun()
    else:
        patterns = st.session_state.patterns
        
        if st.button("Generate Model Recommendations"):
            with st.spinner("Generating model recommendations..."):
                recommendations, error = recommend_model(patterns)
                
                if error:
                    st.error(f"Error generating recommendations: {error}")
                else:
                    st.session_state.recommendations = recommendations
                    
                    st.success("Model recommendations generated!")
                    
                    # Display recommendations
                    st.subheader("🎯 Recommended Models")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"Recommendation {i}: {rec['model']}"):
                            st.write(f"**Model:** {rec['model']}")
                            st.write(f"**Reason:** {rec['reason']}")
                            st.write(f"**Parameters:** {rec['parameters']}")
                    
                    # Model selection guidance
                    st.subheader("📋 Model Selection Guidance")
                    
                    guidance = []
                    
                    if patterns.get('stationarity', {}).get('likely_stationary', False):
                        guidance.append("✅ Data is stationary - ARMA models are suitable")
                    else:
                        guidance.append("⚠️ Data is non-stationary - consider ARIMA with differencing")
                    
                    if patterns.get('seasonality', {}).get('present', False):
                        guidance.append("🔄 Seasonal patterns detected - consider SARIMA models")
                    
                    if patterns.get('trend', {}).get('present', False):
                        guidance.append("📈 Trend present - include differencing or trend terms")
                    
                    for guide in guidance:
                        st.write(guide)
                    
                    # ACF/PACF interpretation for model selection
                    if st.button("Get ACF/PACF Model Suggestions"):
                        with st.spinner("Calculating ACF/PACF for model selection..."):
                            acf_pacf, error = calculate_acf_pacf(data, lags=40)
                            
                            if error:
                                st.error(f"Error calculating ACF/PACF: {error}")
                            else:
                                interpretation, error = interpret_acf_pacf(
                                    acf_pacf['acf'], 
                                    acf_pacf['pacf']
                                )
                                
                                if error:
                                    st.error(f"Error interpreting ACF/PACF: {error}")
                                else:
                                    st.subheader("📊 ACF/PACF Model Suggestions")
                                    st.markdown(interpretation)

with tab5:
    st.subheader("Comprehensive Analysis Report")
    
    st.markdown("""
    **Comprehensive Report** generates a detailed analysis summary including all findings and recommendations.
    """)
    
    if st.button("Generate Comprehensive Report"):
        with st.spinner("Generating comprehensive analysis report..."):
            # Collect all analysis results
            analysis_results = {}
            
            # Get decomposition results
            if 'decomposition' in st.session_state:
                analysis_results['decomposition'] = st.session_state.decomposition
            
            # Get pattern results
            if 'patterns' in st.session_state:
                analysis_results['patterns'] = st.session_state.patterns
            
            # Get stationarity results
            if 'adf_result' in st.session_state:
                analysis_results['adf_result'] = st.session_state.adf_result
            
            if 'kpss_result' in st.session_state:
                analysis_results['kpss_result'] = st.session_state.kpss_result
            
            # Get recommendations
            if 'recommendations' in st.session_state:
                analysis_results['recommendations'] = st.session_state.recommendations
            
            # Generate interpretation
            interpretation, error = generate_analysis_interpretation(
                analysis_results.get('decomposition'),
                analysis_results.get('patterns')
            )
            
            if error:
                st.error(f"Error generating interpretation: {error}")
            else:
                st.success("Comprehensive report generated!")
                
                # Display the report
                st.subheader("📋 Analysis Report")
                st.markdown(interpretation)
                
                # Generate executive summary
                executive_summary, error = generate_executive_summary(analysis_results)
                
                if not error:
                    st.subheader("🎯 Executive Summary")
                    st.markdown(executive_summary)
                
                # Export report
                full_report = []
                full_report.append("# Time Series Analysis Report")
                full_report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                full_report.append("="*60)
                
                # Data overview
                full_report.append("\n## Data Overview")
                full_report.append(f"- Records: {len(data)}")
                full_report.append(f"- Date range: {data.index.min()} to {data.index.max()}")
                full_report.append(f"- Columns: {list(data.columns)}")
                
                # Analysis results
                full_report.append("\n## Analysis Results")
                full_report.append(interpretation)
                
                if not error:
                    full_report.append("\n## Executive Summary")
                    full_report.append(executive_summary)
                
                # Model recommendations
                if 'recommendations' in st.session_state:
                    full_report.append("\n## Model Recommendations")
                    for i, rec in enumerate(st.session_state.recommendations, 1):
                        full_report.append(f"\n### Recommendation {i}: {rec['model']}")
                        full_report.append(f"- **Reason:** {rec['reason']}")
                        full_report.append(f"- **Parameters:** {rec['parameters']}")
                
                report_text = "\n".join(full_report)
                
                # Download button
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report_text,
                    file_name=f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Analysis summary
    st.subheader("📊 Analysis Summary")
    
    completed_analyses = []
    
    if 'adf_result' in st.session_state:
        completed_analyses.append("✅ ADF Stationarity Test")
    
    if 'kpss_result' in st.session_state:
        completed_analyses.append("✅ KPSS Stationarity Test")
    
    if 'decomposition' in st.session_state:
        completed_analyses.append("✅ Time Series Decomposition")
    
    if 'patterns' in st.session_state:
        completed_analyses.append("✅ Pattern Identification")
    
    if 'recommendations' in st.session_state:
        completed_analyses.append("✅ Model Recommendations")
    
    if 'changepoints' in st.session_state:
        completed_analyses.append("✅ Change Point Detection")
    
    if completed_analyses:
        st.success("**Completed Analyses:**")
        for analysis in completed_analyses:
            st.write(analysis)
    else:
        st.info("No analyses completed yet. Use the tabs above to analyze your data.")

# Analysis status
st.markdown("---")
st.subheader("📋 Analysis Status")

if completed_analyses:
    progress = len(completed_analyses) / 6  # Total possible analyses
    st.progress(progress)
    st.write(f"Analysis Progress: {len(completed_analyses)}/6 completed")
    
    # Next steps
    st.subheader("🚀 Next Steps")
    
    next_steps = []
    
    if 'patterns' in st.session_state and 'recommendations' in st.session_state:
        next_steps.append("✅ Ready for model building - visit the **Modeling** page")
    
    if 'decomposition' in st.session_state:
        next_steps.append("📊 Consider visualizing components in the **Visualization** page")
    
    if not next_steps:
        next_steps.append("🔍 Complete more analyses to get personalized recommendations")
    
    for step in next_steps:
        st.write(step)
else:
    st.info("Begin your analysis by selecting one of the tabs above.")
