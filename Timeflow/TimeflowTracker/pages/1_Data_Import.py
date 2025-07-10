import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from utils.data_import import (
    load_csv_file, load_excel_file, load_json_file, 
    connect_to_database, generate_simulated_data,
    validate_time_series_data, prepare_time_series_data,
    detect_data_frequency, initialize_session_state
)
from utils.visualization import create_time_series_plot
from utils.auth import login_required, show_auth_sidebar

# Initialize session state
initialize_session_state()

# Check authentication
login_required()

# Show authentication status in sidebar
show_auth_sidebar()

st.title("📊 Importación de Datos")
st.markdown("Carga tus datos de series temporales desde varias fuentes o genera datos simulados para el análisis.")

# Data source selection
data_source = st.selectbox(
    "Select Data Source",
    ["File Upload", "Database Connection", "Generate Simulated Data"]
)

if data_source == "File Upload":
    st.subheader("📁 File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'json'],
        help="Upload CSV, Excel, or JSON files"
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Loading file..."):
            if file_type == 'csv':
                data, error = load_csv_file(uploaded_file)
            elif file_type in ['xlsx', 'xls']:
                data, error = load_excel_file(uploaded_file)
            elif file_type == 'json':
                data, error = load_json_file(uploaded_file)
            
            if error:
                st.error(f"Error loading file: {error}")
            else:
                st.success("File loaded successfully!")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Data validation
                issues = validate_time_series_data(data)
                if issues:
                    st.warning("Data validation issues found:")
                    for issue in issues:
                        st.write(f"- {issue}")
                
                # Column selection for time series
                st.subheader("Configure Time Series")
                
                # Select date column
                date_columns = []
                for col in data.columns:
                    if data[col].dtype == 'object':
                        try:
                            pd.to_datetime(data[col].head())
                            date_columns.append(col)
                        except:
                            continue
                    elif 'date' in col.lower() or 'time' in col.lower():
                        date_columns.append(col)
                
                if date_columns:
                    date_column = st.selectbox("Select Date Column", date_columns)
                else:
                    st.error("No date column detected. Please ensure your data has a date/time column.")
                    st.stop()
                
                # Select value column
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_columns:
                    value_column = st.selectbox("Select Value Column", numeric_columns)
                else:
                    st.error("No numeric columns detected. Please ensure your data has numeric values.")
                    st.stop()
                
                # Prepare time series data
                if st.button("Prepare Time Series Data"):
                    with st.spinner("Preparing data..."):
                        ts_data, error = prepare_time_series_data(data, date_column, value_column)
                        
                        if error:
                            st.error(f"Error preparing data: {error}")
                        else:
                            st.session_state.data = ts_data
                            st.session_state.original_data = ts_data.copy()
                            
                            # Detect frequency
                            frequency = detect_data_frequency(ts_data)
                            
                            st.success("Time series data prepared successfully!")
                            st.info(f"Detected frequency: {frequency}")
                            
                            # Show prepared data
                            st.subheader("Prepared Time Series Data")
                            st.dataframe(ts_data.head())
                            
                            # Create visualization
                            fig, error = create_time_series_plot(ts_data, "Imported Time Series Data")
                            if error:
                                st.error(f"Error creating plot: {error}")
                            else:
                                st.plotly_chart(fig, use_container_width=True)

elif data_source == "Database Connection":
    st.subheader("🗄️ Database Connection")
    
    db_type = st.selectbox("Database Type", ["PostgreSQL", "SQLite"])
    
    if db_type == "PostgreSQL":
        st.info("Using environment variables for PostgreSQL connection")
        
        # Query input
        query = st.text_area(
            "SQL Query",
            "SELECT * FROM your_table WHERE date_column >= '2020-01-01' ORDER BY date_column",
            help="Enter your SQL query to fetch time series data"
        )
        
        if st.button("Connect and Load Data"):
            with st.spinner("Connecting to database..."):
                connection, error = connect_to_database(db_type, "")
                
                if error:
                    st.error(f"Connection error: {error}")
                else:
                    try:
                        data = pd.read_sql(query, connection)
                        
                        if data.empty:
                            st.warning("Query returned no data")
                        else:
                            st.success("Data loaded from database!")
                            
                            # Show data preview
                            st.subheader("Data Preview")
                            st.dataframe(data.head())
                            
                            # Similar column selection process as file upload
                            st.subheader("Configure Time Series")
                            
                            # Select date column
                            date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                            if not date_columns:
                                date_columns = data.columns.tolist()
                            
                            date_column = st.selectbox("Select Date Column", date_columns)
                            
                            # Select value column
                            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                            value_column = st.selectbox("Select Value Column", numeric_columns)
                            
                            if st.button("Prepare Database Data"):
                                ts_data, error = prepare_time_series_data(data, date_column, value_column)
                                
                                if error:
                                    st.error(f"Error preparing data: {error}")
                                else:
                                    st.session_state.data = ts_data
                                    st.session_state.original_data = ts_data.copy()
                                    
                                    st.success("Database data prepared successfully!")
                                    
                                    # Create visualization
                                    fig, error = create_time_series_plot(ts_data, "Database Time Series Data")
                                    if error:
                                        st.error(f"Error creating plot: {error}")
                                    else:
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
    
    elif db_type == "SQLite":
        st.info("Upload SQLite database file")
        
        db_file = st.file_uploader("Choose SQLite file", type=['db', 'sqlite', 'sqlite3'])
        
        if db_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                tmp_file.write(db_file.read())
                tmp_file_path = tmp_file.name
            
            query = st.text_area(
                "SQL Query",
                "SELECT * FROM your_table ORDER BY date_column",
                help="Enter your SQL query to fetch time series data"
            )
            
            if st.button("Load from SQLite"):
                with st.spinner("Loading data from SQLite..."):
                    try:
                        connection, error = connect_to_database("SQLite", tmp_file_path)
                        
                        if error:
                            st.error(f"Connection error: {error}")
                        else:
                            data = pd.read_sql(query, connection)
                            connection.close()
                            
                            if data.empty:
                                st.warning("Query returned no data")
                            else:
                                st.success("Data loaded from SQLite!")
                                st.dataframe(data.head())
                                
                                # Column selection and preparation similar to other sources
                                
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)

elif data_source == "Generate Simulated Data":
    st.subheader("🎲 Generate Simulated Data")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        num_observations = st.slider(
            "Number of Observations",
            min_value=100,
            max_value=1000,
            value=365,
            step=10
        )
    
    with col2:
        pattern_type = st.selectbox(
            "Pattern Type",
            ["trend_seasonal", "stationary", "random_walk"],
            help="Choose the type of pattern to simulate"
        )
    
    # Pattern descriptions
    pattern_descriptions = {
        "trend_seasonal": "Linear trend with seasonal patterns and noise",
        "stationary": "Stationary series with random variations around a mean",
        "random_walk": "Random walk process with cumulative changes"
    }
    
    st.info(f"**{pattern_type}**: {pattern_descriptions[pattern_type]}")
    
    if st.button("Generate Simulated Data"):
        with st.spinner("Generating simulated data..."):
            sim_data, error = generate_simulated_data(num_observations, pattern_type)
            
            if error:
                st.error(f"Error generating data: {error}")
            else:
                # Prepare as time series
                ts_data, error = prepare_time_series_data(sim_data, 'date', 'value')
                
                if error:
                    st.error(f"Error preparing simulated data: {error}")
                else:
                    st.session_state.data = ts_data
                    st.session_state.original_data = ts_data.copy()
                    
                    st.success("Simulated data generated successfully!")
                    
                    # Show data preview
                    st.subheader("Generated Data Preview")
                    st.dataframe(ts_data.head(10))
                    
                    # Show statistics
                    st.subheader("Data Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean", f"{ts_data.iloc[:, 0].mean():.2f}")
                    with col2:
                        st.metric("Std Dev", f"{ts_data.iloc[:, 0].std():.2f}")
                    with col3:
                        st.metric("Min", f"{ts_data.iloc[:, 0].min():.2f}")
                    with col4:
                        st.metric("Max", f"{ts_data.iloc[:, 0].max():.2f}")
                    
                    # Create visualization
                    fig, error = create_time_series_plot(ts_data, f"Simulated Time Series Data ({pattern_type})")
                    if error:
                        st.error(f"Error creating plot: {error}")
                    else:
                        st.plotly_chart(fig, use_container_width=True)

# Data status
st.markdown("---")
st.subheader("📋 Current Data Status")

if 'data' in st.session_state and st.session_state.data is not None:
    data = st.session_state.data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Data Loaded")
        st.write(f"**Records:** {len(data)}")
    
    with col2:
        st.info("📊 Ready for Analysis")
        st.write(f"**Columns:** {len(data.columns)}")
    
    with col3:
        st.info("🔄 Data Frequency")
        frequency = detect_data_frequency(data)
        st.write(f"**Frequency:** {frequency}")
    
    # Export option
    if st.button("📥 Export Current Data"):
        csv = data.to_csv()
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"time_series_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
else:
    st.warning("No data loaded. Please select a data source and load your data.")
    
    # Quick start options
    st.markdown("### 🚀 Quick Start")
    if st.button("Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            sim_data, error = generate_simulated_data(365, "trend_seasonal")
            
            if not error:
                ts_data, error = prepare_time_series_data(sim_data, 'date', 'value')
                
                if not error:
                    st.session_state.data = ts_data
                    st.session_state.original_data = ts_data.copy()
                    st.rerun()
