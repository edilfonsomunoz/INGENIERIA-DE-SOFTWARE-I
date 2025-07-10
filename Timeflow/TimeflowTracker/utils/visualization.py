import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_time_series_plot(data, title="Time Series Data", interactive=True):
    """Create an interactive time series plot"""
    try:
        fig = go.Figure()
        
        # Add main time series line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.iloc[:, 0] if len(data.columns) > 0 else data,
            mode='lines',
            name='Time Series',
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        if interactive:
            # Add zoom and pan capabilities
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating time series plot: {str(e)}"

def create_outlier_plot(data, outliers, title="Outlier Detection"):
    """Create a plot highlighting outliers"""
    try:
        fig = go.Figure()
        
        # Add normal points
        normal_points = data.loc[~outliers]
        fig.add_trace(go.Scatter(
            x=normal_points.index,
            y=normal_points.iloc[:, 0] if len(normal_points.columns) > 0 else normal_points,
            mode='markers+lines',
            name='Normal',
            marker=dict(color='blue', size=4),
            line=dict(color='blue', width=1)
        ))
        
        # Add outlier points
        outlier_points = data.loc[outliers]
        if len(outlier_points) > 0:
            fig.add_trace(go.Scatter(
                x=outlier_points.index,
                y=outlier_points.iloc[:, 0] if len(outlier_points.columns) > 0 else outlier_points,
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            height=500
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating outlier plot: {str(e)}"

def create_missing_data_plot(data, title="Missing Data Pattern"):
    """Create a plot to visualize missing data patterns"""
    try:
        # Create missing data matrix
        missing_data = data.isnull()
        
        if not missing_data.any().any():
            return None, "No missing data to visualize"
        
        fig = go.Figure()
        
        for col in missing_data.columns:
            missing_indices = missing_data[missing_data[col]].index
            if len(missing_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=missing_indices,
                    y=[col] * len(missing_indices),
                    mode='markers',
                    name=f'Missing in {col}',
                    marker=dict(color='red', size=8, symbol='x')
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Columns',
            height=400
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating missing data plot: {str(e)}"

def create_distribution_plot(data, title="Data Distribution"):
    """Create a histogram and box plot for data distribution"""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Histogram', 'Box Plot'),
            vertical_spacing=0.1
        )
        
        # Get the data series
        series = data.iloc[:, 0] if len(data.columns) > 0 else data
        
        # Add histogram
        fig.add_trace(
            go.Histogram(x=series, name='Distribution', nbinsx=50),
            row=1, col=1
        )
        
        # Add box plot
        fig.add_trace(
            go.Box(y=series, name='Box Plot'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating distribution plot: {str(e)}"

def create_comparison_plot(original_data, processed_data, title="Before vs After Processing"):
    """Create a comparison plot showing original vs processed data"""
    try:
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=original_data.index,
            y=original_data.iloc[:, 0] if len(original_data.columns) > 0 else original_data,
            mode='lines',
            name='Original',
            line=dict(color='lightblue', width=2, dash='dash'),
            opacity=0.7
        ))
        
        # Add processed data
        fig.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data.iloc[:, 0] if len(processed_data.columns) > 0 else processed_data,
            mode='lines',
            name='Processed',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            height=500
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating comparison plot: {str(e)}"

def create_decomposition_plot(trend, seasonal, residual, title="Time Series Decomposition"):
    """Create a decomposition plot showing trend, seasonal, and residual components"""
    try:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.05
        )
        
        # Add trend component
        fig.add_trace(
            go.Scatter(x=trend.index, y=trend.values, name='Trend', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add seasonal component
        fig.add_trace(
            go.Scatter(x=seasonal.index, y=seasonal.values, name='Seasonal', line=dict(color='green')),
            row=2, col=1
        )
        
        # Add residual component
        fig.add_trace(
            go.Scatter(x=residual.index, y=residual.values, name='Residual', line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating decomposition plot: {str(e)}"

def create_correlation_plot(acf_values, pacf_values, lags, title="ACF and PACF"):
    """Create ACF and PACF plots"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)'),
            vertical_spacing=0.1
        )
        
        # Add ACF plot
        fig.add_trace(
            go.Scatter(x=lags, y=acf_values, mode='markers+lines', name='ACF', marker=dict(color='blue')),
            row=1, col=1
        )
        
        # Add confidence intervals for ACF
        confidence_interval = 1.96 / np.sqrt(len(acf_values))
        fig.add_hline(y=confidence_interval, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-confidence_interval, line_dash="dash", line_color="red", row=1, col=1)
        
        # Add PACF plot
        fig.add_trace(
            go.Scatter(x=lags, y=pacf_values, mode='markers+lines', name='PACF', marker=dict(color='green')),
            row=2, col=1
        )
        
        # Add confidence intervals for PACF
        fig.add_hline(y=confidence_interval, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-confidence_interval, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating correlation plot: {str(e)}"

def create_forecast_plot(historical_data, forecast_data, confidence_intervals=None, title="Forecast Results"):
    """Create a forecast plot with confidence intervals"""
    try:
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.iloc[:, 0] if len(historical_data.columns) > 0 else historical_data,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecast data
        fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data.values if hasattr(forecast_data, 'values') else forecast_data,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence intervals if provided
        if confidence_intervals is not None:
            fig.add_trace(go.Scatter(
                x=list(forecast_data.index) + list(forecast_data.index[::-1]),
                y=list(confidence_intervals['upper']) + list(confidence_intervals['lower'][::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            height=500
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating forecast plot: {str(e)}"
