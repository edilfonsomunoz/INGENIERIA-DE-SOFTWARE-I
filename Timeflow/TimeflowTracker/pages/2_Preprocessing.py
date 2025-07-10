import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import (
    detect_outliers, handle_outliers, detect_missing_values,
    interpolate_missing_values, normalize_data, check_data_quality,
    suggest_preprocessing_steps, apply_preprocessing_pipeline
)
from utils.visualization import (
    create_time_series_plot, create_outlier_plot, create_missing_data_plot,
    create_distribution_plot, create_comparison_plot
)
from utils.data_import import initialize_session_state

# Initialize session state
initialize_session_state()

st.title("🔧 Data Preprocessing")
st.markdown("Clean and prepare your time series data for analysis.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No data available. Please load data first using the **Data Import** page.")
    st.stop()

data = st.session_state.data

# Data quality overview
st.subheader("📊 Data Quality Overview")

with st.spinner("Analyzing data quality..."):
    quality_report, error = check_data_quality(data)
    
    if error:
        st.error(f"Error analyzing data quality: {error}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", quality_report['total_records'])
        
        with col2:
            st.metric("Missing Values", quality_report['missing_values'])
        
        with col3:
            st.metric("Duplicate Dates", quality_report['duplicate_dates'])
        
        with col4:
            time_span = quality_report['date_range']['span_days']
            st.metric("Time Span (Days)", time_span)

# Preprocessing steps
st.subheader("🔄 Preprocessing Steps")

# Get preprocessing suggestions
suggestions, error = suggest_preprocessing_steps(data)
if not error and suggestions:
    st.info("**Recommended preprocessing steps:**")
    for suggestion in suggestions:
        st.write(f"- {suggestion['step']}: {suggestion['reason']}")

# Create tabs for different preprocessing operations
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Outlier Detection", "📈 Missing Values", "📊 Normalization", "🔍 Quality Check"])

with tab1:
    st.subheader("Outlier Detection and Handling")
    
    # Outlier detection method selection
    outlier_method = st.selectbox(
        "Outlier Detection Method",
        ["iqr", "zscore", "modified_zscore"],
        help="Choose the method for detecting outliers"
    )
    
    # Threshold for z-score methods
    if outlier_method in ["zscore", "modified_zscore"]:
        threshold = st.slider("Threshold", 1.0, 5.0, 3.0, 0.1)
    else:
        threshold = 3.0
    
    if st.button("Detect Outliers"):
        with st.spinner("Detecting outliers..."):
            outliers, error = detect_outliers(data.iloc[:, 0], outlier_method, threshold)
            
            if error:
                st.error(f"Error detecting outliers: {error}")
            else:
                st.session_state.outliers = outliers
                
                outlier_count = outliers.sum()
                st.info(f"Found {outlier_count} outliers ({outlier_count/len(data)*100:.1f}% of data)")
                
                # Visualize outliers
                fig, error = create_outlier_plot(data, outliers)
                if error:
                    st.error(f"Error creating outlier plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show outlier values
                if outlier_count > 0:
                    st.subheader("Outlier Values")
                    outlier_data = data.loc[outliers]
                    st.dataframe(outlier_data)
    
    # Outlier handling
    if 'outliers' in st.session_state:
        st.subheader("Outlier Handling")
        
        outlier_treatment = st.selectbox(
            "Outlier Treatment Method",
            ["replace_median", "replace_mean", "remove", "cap"],
            help="Choose how to handle detected outliers"
        )
        
        if st.button("Apply Outlier Treatment"):
            with st.spinner("Handling outliers..."):
                cleaned_data, error = handle_outliers(data.iloc[:, 0], st.session_state.outliers, outlier_treatment)
                
                if error:
                    st.error(f"Error handling outliers: {error}")
                else:
                    # Create a new DataFrame with cleaned data
                    cleaned_df = data.copy()
                    cleaned_df.iloc[:, 0] = cleaned_data
                    
                    st.session_state.processed_data = cleaned_df
                    
                    st.success("Outliers handled successfully!")
                    
                    # Show comparison
                    fig, error = create_comparison_plot(data, cleaned_df, "Original vs Outlier-Cleaned Data")
                    if error:
                        st.error(f"Error creating comparison plot: {error}")
                    else:
                        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Missing Values")
    
    # Detect missing values
    missing_info, error = detect_missing_values(data)
    
    if error:
        st.error(f"Error detecting missing values: {error}")
    else:
        if missing_info['has_missing']:
            st.warning(f"Found {missing_info['total_missing']} missing values")
            
            # Show missing data pattern
            fig, error = create_missing_data_plot(data)
            if error and "No missing data" not in error:
                st.error(f"Error creating missing data plot: {error}")
            elif not error:
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpolation method selection
            interpolation_method = st.selectbox(
                "Interpolation Method",
                ["linear", "polynomial", "spline", "forward_fill", "backward_fill", "mean", "median"],
                help="Choose the method for interpolating missing values"
            )
            
            if st.button("Interpolate Missing Values"):
                with st.spinner("Interpolating missing values..."):
                    interpolated_data, error = interpolate_missing_values(data, interpolation_method)
                    
                    if error:
                        st.error(f"Error interpolating missing values: {error}")
                    else:
                        st.session_state.processed_data = interpolated_data
                        
                        st.success("Missing values interpolated successfully!")
                        
                        # Show comparison
                        fig, error = create_comparison_plot(data, interpolated_data, "Original vs Interpolated Data")
                        if error:
                            st.error(f"Error creating comparison plot: {error}")
                        else:
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected in the dataset")

with tab3:
    st.subheader("Data Normalization")
    
    # Show current data distribution
    fig, error = create_distribution_plot(data, "Original Data Distribution")
    if error:
        st.error(f"Error creating distribution plot: {error}")
    else:
        st.plotly_chart(fig, use_container_width=True)
    
    # Normalization method selection
    normalization_method = st.selectbox(
        "Normalization Method",
        ["minmax", "zscore", "robust", "log"],
        help="Choose the normalization method"
    )
    
    # Method descriptions
    method_descriptions = {
        "minmax": "Scale features to a fixed range [0,1]",
        "zscore": "Standardize features to have mean=0 and std=1",
        "robust": "Use median and IQR, robust to outliers",
        "log": "Apply logarithmic transformation"
    }
    
    st.info(f"**{normalization_method}**: {method_descriptions[normalization_method]}")
    
    if st.button("Apply Normalization"):
        with st.spinner("Normalizing data..."):
            normalized_data, scaler, error = normalize_data(data.iloc[:, 0], normalization_method)
            
            if error:
                st.error(f"Error normalizing data: {error}")
            else:
                # Create normalized DataFrame
                normalized_df = data.copy()
                normalized_df.iloc[:, 0] = normalized_data
                
                st.session_state.processed_data = normalized_df
                st.session_state.scaler = scaler
                
                st.success("Data normalized successfully!")
                
                # Show comparison
                fig, error = create_comparison_plot(data, normalized_df, "Original vs Normalized Data")
                if error:
                    st.error(f"Error creating comparison plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show normalized distribution
                fig, error = create_distribution_plot(normalized_df, "Normalized Data Distribution")
                if error:
                    st.error(f"Error creating distribution plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Data Quality Check")
    
    if quality_report:
        st.json(quality_report)
    
    # Additional quality checks
    if st.button("Run Comprehensive Quality Check"):
        with st.spinner("Running quality checks..."):
            # Check for duplicates
            duplicates = data.duplicated().sum()
            
            # Check for constant values
            constant_cols = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].nunique() <= 1:
                    constant_cols.append(col)
            
            # Check for high correlation (if multiple columns)
            high_corr_pairs = []
            if len(data.columns) > 1:
                corr_matrix = data.corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.95:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Display results
            st.subheader("Quality Check Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duplicate Rows", duplicates)
                if duplicates > 0:
                    st.warning("Consider removing duplicate rows")
            
            with col2:
                st.metric("Constant Columns", len(constant_cols))
                if constant_cols:
                    st.warning(f"Constant columns: {constant_cols}")
            
            with col3:
                st.metric("High Correlation Pairs", len(high_corr_pairs))
                if high_corr_pairs:
                    st.warning(f"High correlation pairs: {high_corr_pairs}")

# Preprocessing pipeline
st.subheader("🔧 Preprocessing Pipeline")

if st.button("Apply Automated Preprocessing"):
    with st.spinner("Applying preprocessing pipeline..."):
        # Define preprocessing steps
        steps = []
        
        # Check for missing values
        missing_info, _ = detect_missing_values(data)
        if missing_info and missing_info['has_missing']:
            steps.append({
                'action': 'interpolate_missing',
                'interpolation_method': 'linear'
            })
        
        # Check for outliers
        outliers, _ = detect_outliers(data.iloc[:, 0])
        if outliers is not None and outliers.sum() > 0:
            steps.append({
                'action': 'handle_outliers',
                'outlier_method': 'iqr',
                'outlier_treatment': 'replace_median'
            })
        
        # Apply normalization if data has high variance
        if data.iloc[:, 0].std() > 1000:
            steps.append({
                'action': 'normalize',
                'normalization_method': 'zscore'
            })
        
        if steps:
            processed_data, processing_log, error = apply_preprocessing_pipeline(data, steps)
            
            if error:
                st.error(f"Error in preprocessing pipeline: {error}")
            else:
                st.session_state.processed_data = processed_data
                
                st.success("Preprocessing pipeline completed!")
                
                # Show processing log
                st.subheader("Processing Log")
                for log_entry in processing_log:
                    st.write(f"✅ {log_entry}")
                
                # Show comparison
                fig, error = create_comparison_plot(data, processed_data, "Original vs Processed Data")
                if error:
                    st.error(f"Error creating comparison plot: {error}")
                else:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No preprocessing steps needed - data appears to be clean!")

# Current preprocessing status
st.markdown("---")
st.subheader("📋 Preprocessing Status")

if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    processed_data = st.session_state.processed_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ Processed Data Available")
        st.write(f"**Records:** {len(processed_data)}")
        
        # Show processed data preview
        st.subheader("Processed Data Preview")
        st.dataframe(processed_data.head())
    
    with col2:
        st.info("📊 Ready for Analysis")
        st.write(f"**Columns:** {len(processed_data.columns)}")
        
        # Update main data
        if st.button("Use Processed Data for Analysis"):
            st.session_state.data = processed_data
            st.success("Processed data is now active for analysis!")
            st.rerun()
    
    # Export processed data
    if st.button("📥 Export Processed Data"):
        csv = processed_data.to_csv()
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
else:
    st.info("No processed data available. Apply preprocessing steps above to clean your data.")
