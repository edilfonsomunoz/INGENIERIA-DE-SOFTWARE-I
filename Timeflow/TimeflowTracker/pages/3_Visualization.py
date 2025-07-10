import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.visualization import (
    create_time_series_plot, create_distribution_plot, create_comparison_plot,
    create_decomposition_plot, create_correlation_plot, create_forecast_plot
)
from utils.time_series_analysis import decompose_time_series, calculate_acf_pacf
from utils.data_import import initialize_session_state

# Initialize session state
initialize_session_state()

st.title("📊 Interactive Visualization")
st.markdown("Explore your time series data with interactive charts and visualizations.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No data available. Please load data first using the **Data Import** page.")
    st.stop()

data = st.session_state.data

# Main visualization options
st.subheader("🎨 Visualization Options")

# Create tabs for different visualization types
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series Plot", 
    "📊 Distribution Analysis", 
    "🔍 Decomposition", 
    "📉 Correlation Analysis",
    "🔄 Comparison Views"
])

with tab1:
    st.subheader("Time Series Plot")
    
    # Plot customization options
    col1, col2 = st.columns(2)
    
    with col1:
        chart_title = st.text_input("Chart Title", "Time Series Data")
        show_range_selector = st.checkbox("Show Range Selector", value=True)
    
    with col2:
        line_color = st.color_picker("Line Color", "#1f77b4")
        line_width = st.slider("Line Width", 1, 5, 2)
    
    # Create the plot
    if st.button("Generate Time Series Plot"):
        with st.spinner("Creating time series plot..."):
            fig, error = create_time_series_plot(data, chart_title, interactive=show_range_selector)
            
            if error:
                st.error(f"Error creating plot: {error}")
            else:
                # Customize the plot
                fig.update_traces(line=dict(color=line_color, width=line_width))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                st.subheader("📥 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Download as PNG"):
                        fig.write_image("time_series_plot.png")
                        st.success("Plot saved as PNG!")
                
                with col2:
                    if st.button("Download as HTML"):
                        fig.write_html("time_series_plot.html")
                        st.success("Plot saved as HTML!")
                
                with col3:
                    if st.button("Download as SVG"):
                        fig.write_image("time_series_plot.svg")
                        st.success("Plot saved as SVG!")
    
    # Statistical overlays
    st.subheader("📊 Statistical Overlays")
    
    overlays = st.multiselect(
        "Select overlays to add:",
        ["Moving Average", "Bollinger Bands", "Trend Line", "Confidence Intervals"]
    )
    
    if overlays and st.button("Add Statistical Overlays"):
        with st.spinner("Adding statistical overlays..."):
            fig = go.Figure()
            
            # Add main time series
            series = data.iloc[:, 0] if len(data.columns) > 0 else data
            fig.add_trace(go.Scatter(
                x=data.index,
                y=series,
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=2)
            ))
            
            # Add overlays
            if "Moving Average" in overlays:
                window = st.sidebar.slider("Moving Average Window", 5, 50, 20)
                ma = series.rolling(window=window).mean()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=ma,
                    mode='lines',
                    name=f'MA({window})',
                    line=dict(color='red', width=1, dash='dash')
                ))
            
            if "Bollinger Bands" in overlays:
                window = st.sidebar.slider("Bollinger Bands Window", 10, 50, 20)
                std_dev = st.sidebar.slider("Standard Deviations", 1.0, 3.0, 2.0)
                
                ma = series.rolling(window=window).mean()
                std = series.rolling(window=window).std()
                
                upper_band = ma + (std * std_dev)
                lower_band = ma - (std * std_dev)
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=upper_band,
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=lower_band,
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ))
            
            if "Trend Line" in overlays:
                # Calculate trend line
                x_numeric = np.arange(len(series))
                coeffs = np.polyfit(x_numeric, series.dropna(), 1)
                trend_line = np.polyval(coeffs, x_numeric)
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='green', width=2, dash='dot')
                ))
            
            fig.update_layout(
                title="Time Series with Statistical Overlays",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Distribution Analysis")
    
    # Distribution plot
    fig, error = create_distribution_plot(data, "Data Distribution Analysis")
    
    if error:
        st.error(f"Error creating distribution plot: {error}")
    else:
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    
    series = data.iloc[:, 0] if len(data.columns) > 0 else data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{series.mean():.2f}")
        st.metric("Median", f"{series.median():.2f}")
    
    with col2:
        st.metric("Std Dev", f"{series.std():.2f}")
        st.metric("Variance", f"{series.var():.2f}")
    
    with col3:
        st.metric("Skewness", f"{series.skew():.2f}")
        st.metric("Kurtosis", f"{series.kurtosis():.2f}")
    
    with col4:
        st.metric("Min", f"{series.min():.2f}")
        st.metric("Max", f"{series.max():.2f}")
    
    # Quantile analysis
    st.subheader("📈 Quantile Analysis")
    
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_values = [series.quantile(q) for q in quantiles]
    
    quantile_df = pd.DataFrame({
        'Quantile': [f"{q*100:.0f}%" for q in quantiles],
        'Value': quantile_values
    })
    
    st.dataframe(quantile_df, use_container_width=True)
    
    # Box plot details
    st.subheader("📦 Box Plot Analysis")
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Q1 (25th percentile)", f"{Q1:.2f}")
        st.metric("Q3 (75th percentile)", f"{Q3:.2f}")
    
    with col2:
        st.metric("IQR", f"{IQR:.2f}")
        st.metric("Outlier Threshold", f"{1.5 * IQR:.2f}")

with tab3:
    st.subheader("Time Series Decomposition")
    
    # Decomposition parameters
    col1, col2 = st.columns(2)
    
    with col1:
        decomp_model = st.selectbox(
            "Decomposition Model",
            ["additive", "multiplicative"],
            help="Choose the decomposition model type"
        )
    
    with col2:
        period = st.number_input(
            "Seasonal Period",
            min_value=2,
            max_value=min(len(data)//2, 365),
            value=12,
            help="Enter the seasonal period (e.g., 12 for monthly, 7 for weekly)"
        )
    
    if st.button("Perform Decomposition"):
        with st.spinner("Performing time series decomposition..."):
            decomposition, error = decompose_time_series(data, model=decomp_model, period=period)
            
            if error:
                st.error(f"Error in decomposition: {error}")
            else:
                st.success("Decomposition completed successfully!")
                
                # Store decomposition results
                st.session_state.decomposition = decomposition
                
                # Create decomposition plot
                fig, error = create_decomposition_plot(
                    decomposition['trend'],
                    decomposition['seasonal'],
                    decomposition['residual'],
                    f"Time Series Decomposition ({decomp_model})"
                )
                
                if error:
                    st.error(f"Error creating decomposition plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Component statistics
                st.subheader("📊 Component Statistics")
                
                components = ['trend', 'seasonal', 'residual']
                component_stats = {}
                
                for comp in components:
                    comp_data = decomposition[comp].dropna()
                    if len(comp_data) > 0:
                        component_stats[comp] = {
                            'mean': comp_data.mean(),
                            'std': comp_data.std(),
                            'min': comp_data.min(),
                            'max': comp_data.max()
                        }
                
                stats_df = pd.DataFrame(component_stats).T
                st.dataframe(stats_df)
                
                # Residual analysis
                st.subheader("🔍 Residual Analysis")
                
                residuals = decomposition['residual'].dropna()
                
                if len(residuals) > 0:
                    # Residual distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=residuals,
                        nbinsx=30,
                        name='Residuals',
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title="Residual Distribution",
                        xaxis_title="Residual Value",
                        yaxis_title="Frequency",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residual statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Residual Mean", f"{residuals.mean():.4f}")
                    
                    with col2:
                        st.metric("Residual Std", f"{residuals.std():.4f}")
                    
                    with col3:
                        st.metric("Residual Range", f"{residuals.max() - residuals.min():.4f}")

with tab4:
    st.subheader("Correlation Analysis (ACF/PACF)")
    
    # ACF/PACF parameters
    max_lags = st.slider(
        "Number of Lags",
        min_value=10,
        max_value=min(len(data)//4, 100),
        value=40,
        help="Maximum number of lags to calculate"
    )
    
    if st.button("Calculate ACF and PACF"):
        with st.spinner("Calculating autocorrelation functions..."):
            acf_pacf_result, error = calculate_acf_pacf(data, lags=max_lags)
            
            if error:
                st.error(f"Error calculating ACF/PACF: {error}")
            else:
                st.success("ACF and PACF calculated successfully!")
                
                # Store results
                st.session_state.acf_pacf = acf_pacf_result
                
                # Create ACF/PACF plot
                fig, error = create_correlation_plot(
                    acf_pacf_result['acf'],
                    acf_pacf_result['pacf'],
                    list(acf_pacf_result['lags']),
                    "Autocorrelation and Partial Autocorrelation Functions"
                )
                
                if error:
                    st.error(f"Error creating correlation plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Significant lags analysis
                st.subheader("📊 Significant Lags Analysis")
                
                # Find significant lags (threshold of 0.2)
                significant_acf = []
                significant_pacf = []
                
                for i, (acf_val, pacf_val) in enumerate(zip(acf_pacf_result['acf'][1:], acf_pacf_result['pacf'][1:])):
                    if abs(acf_val) > 0.2:
                        significant_acf.append(i + 1)
                    if abs(pacf_val) > 0.2:
                        significant_pacf.append(i + 1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Significant ACF Lags:**")
                    if significant_acf:
                        st.write(f"Lags: {significant_acf[:10]}")  # Show first 10
                    else:
                        st.write("No significant ACF lags found")
                
                with col2:
                    st.write("**Significant PACF Lags:**")
                    if significant_pacf:
                        st.write(f"Lags: {significant_pacf[:10]}")  # Show first 10
                    else:
                        st.write("No significant PACF lags found")
                
                # Model suggestions based on ACF/PACF
                st.subheader("🎯 Model Suggestions")
                
                suggestions = []
                
                if significant_pacf and not significant_acf:
                    suggestions.append(f"Consider AR({min(significant_pacf[0], 3)}) model")
                elif significant_acf and not significant_pacf:
                    suggestions.append(f"Consider MA({min(significant_acf[0], 3)}) model")
                elif significant_acf and significant_pacf:
                    suggestions.append(f"Consider ARMA({min(significant_pacf[0], 3)}, {min(significant_acf[0], 3)}) model")
                else:
                    suggestions.append("Data may be white noise or require differencing")
                
                for suggestion in suggestions:
                    st.info(suggestion)
                
                # Export ACF/PACF data
                if st.button("📥 Export ACF/PACF Data"):
                    acf_pacf_df = pd.DataFrame({
                        'Lag': list(acf_pacf_result['lags']),
                        'ACF': acf_pacf_result['acf'],
                        'PACF': acf_pacf_result['pacf']
                    })
                    
                    csv = acf_pacf_df.to_csv(index=False)
                    st.download_button(
                        label="Download ACF/PACF as CSV",
                        data=csv,
                        file_name=f"acf_pacf_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

with tab5:
    st.subheader("Comparison Views")
    
    # Check if processed data is available
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        processed_data = st.session_state.processed_data
        
        # Original vs Processed comparison
        if st.button("Compare Original vs Processed Data"):
            with st.spinner("Creating comparison plot..."):
                fig, error = create_comparison_plot(
                    data, 
                    processed_data, 
                    "Original vs Processed Data Comparison"
                )
                
                if error:
                    st.error(f"Error creating comparison plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics comparison
                st.subheader("📊 Statistics Comparison")
                
                original_stats = data.iloc[:, 0].describe()
                processed_stats = processed_data.iloc[:, 0].describe()
                
                comparison_df = pd.DataFrame({
                    'Original': original_stats,
                    'Processed': processed_stats,
                    'Difference': processed_stats - original_stats
                })
                
                st.dataframe(comparison_df)
    else:
        st.info("No processed data available for comparison. Visit the **Preprocessing** page to process your data first.")
    
    # Custom comparison
    st.subheader("🔧 Custom Period Comparison")
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=data.index.min().date(),
            min_value=data.index.min().date(),
            max_value=data.index.max().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=data.index.max().date(),
            min_value=data.index.min().date(),
            max_value=data.index.max().date()
        )
    
    if st.button("Compare Time Periods"):
        if start_date >= end_date:
            st.error("End date must be after start date")
        else:
            # Filter data for the selected period
            mask = (data.index.date >= start_date) & (data.index.date <= end_date)
            filtered_data = data.loc[mask]
            
            if len(filtered_data) == 0:
                st.warning("No data available for the selected period")
            else:
                # Create comparison plot
                fig, error = create_time_series_plot(
                    filtered_data,
                    f"Time Series Data ({start_date} to {end_date})"
                )
                
                if error:
                    st.error(f"Error creating plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Period statistics
                st.subheader("📊 Period Statistics")
                
                series = filtered_data.iloc[:, 0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Records", len(filtered_data))
                    st.metric("Mean", f"{series.mean():.2f}")
                
                with col2:
                    st.metric("Std Dev", f"{series.std():.2f}")
                    st.metric("Min", f"{series.min():.2f}")
                
                with col3:
                    st.metric("Max", f"{series.max():.2f}")
                    st.metric("Range", f"{series.max() - series.min():.2f}")
                
                with col4:
                    st.metric("Skewness", f"{series.skew():.2f}")
                    st.metric("Kurtosis", f"{series.kurtosis():.2f}")

# Visualization summary
st.markdown("---")
st.subheader("📋 Visualization Summary")

# Check what visualizations have been created
visualizations_created = []

if 'decomposition' in st.session_state:
    visualizations_created.append("✅ Time Series Decomposition")

if 'acf_pacf' in st.session_state:
    visualizations_created.append("✅ ACF/PACF Analysis")

if visualizations_created:
    st.success("Completed Visualizations:")
    for viz in visualizations_created:
        st.write(viz)
else:
    st.info("No visualizations created yet. Use the tabs above to explore your data.")

# Export options
st.subheader("📥 Export All Visualizations")

if st.button("Generate Visualization Report"):
    with st.spinner("Generating visualization report..."):
        # Create a comprehensive report
        report = []
        
        report.append("# Time Series Visualization Report")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*50)
        
        # Data overview
        report.append("\n## Data Overview")
        report.append(f"- Total records: {len(data)}")
        report.append(f"- Date range: {data.index.min()} to {data.index.max()}")
        report.append(f"- Time span: {(data.index.max() - data.index.min()).days} days")
        
        # Basic statistics
        series = data.iloc[:, 0]
        report.append(f"\n## Basic Statistics")
        report.append(f"- Mean: {series.mean():.2f}")
        report.append(f"- Standard Deviation: {series.std():.2f}")
        report.append(f"- Minimum: {series.min():.2f}")
        report.append(f"- Maximum: {series.max():.2f}")
        
        # Visualization results
        if 'decomposition' in st.session_state:
            report.append(f"\n## Decomposition Analysis")
            report.append(f"- Model: {st.session_state.decomposition['model']}")
            report.append(f"- Period: {st.session_state.decomposition['period']}")
        
        if 'acf_pacf' in st.session_state:
            report.append(f"\n## Correlation Analysis")
            report.append(f"- ACF/PACF calculated for {len(st.session_state.acf_pacf['acf'])} lags")
        
        report_text = "\n".join(report)
        
        st.download_button(
            label="Download Visualization Report",
            data=report_text,
            file_name=f"visualization_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        st.success("Visualization report generated!")
