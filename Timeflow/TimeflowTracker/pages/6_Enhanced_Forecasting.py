import streamlit as st
import pandas as pd
import numpy as np
from utils.advanced_models import generate_model_forecasts
from utils.forecasting import (
    generate_forecasts, calculate_forecast_metrics, backtest_model,
    cross_validate_model, forecast_scenarios, export_forecasts,
    calculate_forecast_intervals, forecast_diagnostics
)
from utils.visualization import create_forecast_plot, create_time_series_plot
from utils.interpretation import (
    generate_forecast_interpretation, generate_comprehensive_report
)
from utils.pdf_export import create_downloadable_pdf
from utils.data_import import initialize_session_state
from utils.auth import login_required, show_auth_sidebar
import plotly.graph_objects as go
from datetime import datetime

# Initialize session state
initialize_session_state()

# Check authentication
login_required()

# Show authentication status in sidebar
show_auth_sidebar()

st.title("🔮 Pronóstico Avanzado")
st.markdown("Genera pronósticos y guarda todo el análisis en PDF utilizando tus modelos ajustados.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No hay datos disponibles. Por favor carga datos primero usando la página de **Importación de Datos**.")
    st.stop()

data = st.session_state.data

# Check for available models
available_models = {}

# Check for traditional models
if 'manual_model' in st.session_state:
    available_models['Modelo SARIMA Manual'] = st.session_state.manual_model

if 'auto_model' in st.session_state:
    auto_result = st.session_state.auto_model
    available_models['Modelo SARIMA Auto'] = {
        'model': auto_result['best_model'],
        'order': auto_result['best_order'],
        'seasonal_order': auto_result['best_seasonal_order'],
        'aic': auto_result['best_aic'],
        'bic': auto_result['best_model'].bic,
        'fitted_values': auto_result['fitted_values'],
        'residuals': auto_result['residuals']
    }

# Check for advanced models
if 'advanced_models' in st.session_state:
    for name, model in st.session_state.advanced_models.items():
        available_models[f'Modelo {name}'] = model

if len(available_models) == 0:
    st.warning("⚠️ No hay modelos disponibles. Por favor ajusta al menos un modelo usando las páginas de **Modelado**.")
    st.stop()

# Initialize forecast results in session state
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}

# Create tabs for forecasting workflow
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Generar Pronósticos", 
    "📊 Validación", 
    "📈 Análisis Avanzado",
    "📄 Exportar PDF"
])

with tab1:
    st.subheader("Generar Pronósticos")
    
    # Model selection
    selected_model_name = st.selectbox(
        "Selecciona el modelo para pronóstico:",
        list(available_models.keys())
    )
    selected_model = available_models[selected_model_name]
    
    # Display model information
    st.info(f"**Modelo Seleccionado:** {selected_model_name}")
    if 'order' in selected_model:
        st.info(f"**Especificación:** SARIMA{selected_model['order']}×{selected_model.get('seasonal_order', 'N/A')}")
    elif 'model_type' in selected_model:
        st.info(f"**Tipo:** {selected_model['model_type']}")
    
    # Forecast parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_steps = st.number_input(
            "Horizonte de Pronóstico (períodos)",
            min_value=1,
            max_value=365,
            value=12,
            help="Número de períodos a pronosticar hacia el futuro"
        )
    
    with col2:
        confidence_level = st.selectbox(
            "Nivel de Confianza",
            [0.80, 0.90, 0.95, 0.99],
            index=2,
            help="Nivel de confianza para intervalos de pronóstico"
        )
    
    # Generate forecasts
    if st.button("Generar Pronósticos"):
        with st.spinner("Generando pronósticos..."):
            try:
                # Check if it's an advanced model
                if 'model_type' in selected_model:
                    forecast_result, error = generate_model_forecasts(selected_model, steps=forecast_steps)
                    if error:
                        st.error(f"Error generando pronósticos: {error}")
                    else:
                        st.session_state.forecast_results = {
                            'forecasts': forecast_result,
                            'model_name': selected_model_name,
                            'model_type': selected_model['model_type'],
                            'steps': forecast_steps
                        }
                        st.success("¡Pronósticos generados exitosamente!")
                else:
                    # Traditional SARIMA model
                    forecast_result, error = generate_forecasts(selected_model, steps=forecast_steps)
                    if error:
                        st.error(f"Error generando pronósticos: {error}")
                    else:
                        st.session_state.forecast_results = {
                            'forecasts': forecast_result['forecasts'],
                            'model_name': selected_model_name,
                            'model_type': 'SARIMA',
                            'steps': forecast_steps
                        }
                        st.success("¡Pronósticos generados exitosamente!")
                
                # Display forecast results if successful
                if 'forecast_results' in st.session_state and st.session_state.forecast_results:
                    st.subheader("📊 Resultados del Pronóstico")
                    
                    forecasts_df = st.session_state.forecast_results['forecasts']
                    
                    # Show forecast table
                    st.write("**Tabla de Pronósticos:**")
                    st.dataframe(forecasts_df, use_container_width=True)
                    
                    # Create forecast plot
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data.iloc[:, 0],
                        mode='lines',
                        name='Datos Históricos',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add forecast
                    forecast_dates = pd.date_range(
                        start=data.index[-1],
                        periods=forecast_steps + 1,
                        freq='D'
                    )[1:]
                    
                    if 'forecast' in forecasts_df.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecasts_df['forecast'],
                            mode='lines+markers',
                            name='Pronóstico',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Add confidence intervals if available
                        if 'lower_ci' in forecasts_df.columns and 'upper_ci' in forecasts_df.columns:
                            fig.add_trace(go.Scatter(
                                x=list(forecast_dates) + list(forecast_dates[::-1]),
                                y=list(forecasts_df['upper_ci']) + list(forecasts_df['lower_ci'][::-1]),
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name=f'IC {int(confidence_level*100)}%'
                            ))
                    else:
                        # For volatility forecasts (GARCH/EGARCH)
                        if 'volatility_forecast' in forecasts_df.columns:
                            fig.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=forecasts_df['volatility_forecast'],
                                mode='lines+markers',
                                name='Pronóstico de Volatilidad',
                                line=dict(color='red', width=2)
                            ))
                    
                    fig.update_layout(
                        title=f'Pronóstico - {selected_model_name}',
                        xaxis_title='Fecha',
                        yaxis_title='Valor',
                        hovermode='x unified',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast statistics
                    st.subheader("📈 Estadísticas del Pronóstico")
                    
                    if 'forecast' in forecasts_df.columns:
                        forecast_values = forecasts_df['forecast']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Pronóstico Promedio", f"{forecast_values.mean():.2f}")
                        
                        with col2:
                            st.metric("Desviación Estándar", f"{forecast_values.std():.2f}")
                        
                        with col3:
                            st.metric("Pronóstico Mínimo", f"{forecast_values.min():.2f}")
                        
                        with col4:
                            st.metric("Pronóstico Máximo", f"{forecast_values.max():.2f}")
                        
                        # Forecast trend
                        if len(forecast_values) > 1:
                            trend = "Creciente" if forecast_values.iloc[-1] > forecast_values.iloc[0] else "Decreciente"
                            st.info(f"**Tendencia del Pronóstico:** {trend}")
            
            except Exception as e:
                st.error(f"Error inesperado: {str(e)}")

with tab2:
    st.subheader("Validación del Modelo")
    
    st.markdown("""
    **Validación del Modelo** evalúa el rendimiento del pronóstico usando datos históricos.
    """)
    
    # Validation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Tamaño de Prueba (proporción)",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proporción de datos para prueba"
        )
    
    with col2:
        validation_model_name = st.selectbox(
            "Selecciona modelo a validar:",
            list(available_models.keys()),
            key="validation_model"
        )
    
    # Backtesting
    if st.button("Ejecutar Validación"):
        with st.spinner("Ejecutando validación..."):
            validation_model = available_models[validation_model_name]
            
            # Split data for validation
            split_point = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_point]
            test_data = data.iloc[split_point:]
            
            try:
                # Generate forecasts for test period
                if 'model_type' in validation_model:
                    forecast_result, error = generate_model_forecasts(validation_model, steps=len(test_data))
                else:
                    forecast_result, error = generate_forecasts(validation_model, steps=len(test_data))
                
                if error:
                    st.error(f"Error en validación: {error}")
                else:
                    # Calculate metrics
                    if 'forecast' in forecast_result.columns:
                        actual = test_data.iloc[:, 0].values
                        predicted = forecast_result['forecast'].values[:len(actual)]
                        
                        # Calculate metrics
                        mse = np.mean((actual - predicted) ** 2)
                        mae = np.mean(np.abs(actual - predicted))
                        rmse = np.sqrt(mse)
                        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                        
                        st.success("¡Validación completada!")
                        
                        # Display metrics
                        st.write("**Métricas de Precisión:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col2:
                            st.metric("MAE", f"{mae:.4f}")
                        with col3:
                            st.metric("MAPE", f"{mape:.2f}%")
                        with col4:
                            st.metric("MSE", f"{mse:.4f}")
                        
                        # Interpretation
                        st.subheader("📝 Interpretación de Métricas")
                        
                        if mape < 10:
                            st.success("✅ Excelente precisión de pronóstico (MAPE < 10%)")
                        elif mape < 20:
                            st.info("✅ Buena precisión de pronóstico (MAPE < 20%)")
                        elif mape < 50:
                            st.warning("⚠️ Precisión moderada de pronóstico (MAPE < 50%)")
                        else:
                            st.error("❌ Precisión pobre de pronóstico (MAPE > 50%)")
                        
                        # Plot validation results
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=train_data.index,
                            y=train_data.iloc[:, 0],
                            mode='lines',
                            name='Entrenamiento',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=test_data.index,
                            y=test_data.iloc[:, 0],
                            mode='lines',
                            name='Datos Reales',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=test_data.index[:len(predicted)],
                            y=predicted,
                            mode='lines',
                            name='Pronóstico',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title='Validación del Modelo - Datos Reales vs Pronóstico',
                            xaxis_title='Fecha',
                            yaxis_title='Valor',
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store validation results
                        st.session_state.validation_results = {
                            'metrics': {
                                'rmse': rmse,
                                'mae': mae,
                                'mape': mape,
                                'mse': mse
                            },
                            'actual': actual,
                            'predicted': predicted,
                            'model_name': validation_model_name
                        }
            
            except Exception as e:
                st.error(f"Error en validación: {str(e)}")

with tab3:
    st.subheader("Análisis Avanzado de Pronósticos")
    
    if 'forecast_results' not in st.session_state or not st.session_state.forecast_results:
        st.info("Por favor genera pronósticos primero en la pestaña **Generar Pronósticos**.")
    else:
        st.markdown("### 📊 Resumen del Análisis")
        
        forecast_data = st.session_state.forecast_results
        
        # Summary information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modelo Utilizado", forecast_data['model_name'])
        with col2:
            st.metric("Tipo de Modelo", forecast_data['model_type'])
        with col3:
            st.metric("Períodos Pronosticados", forecast_data['steps'])
        
        # Advanced analysis
        st.markdown("### 🔍 Análisis Detallado")
        
        forecasts_df = forecast_data['forecasts']
        
        if 'forecast' in forecasts_df.columns:
            forecast_values = forecasts_df['forecast']
            
            # Statistical analysis
            st.write("**Análisis Estadístico:**")
            
            stats_data = {
                'Métrica': ['Media', 'Mediana', 'Desviación Estándar', 'Varianza', 'Mínimo', 'Máximo', 'Rango'],
                'Valor': [
                    f"{forecast_values.mean():.4f}",
                    f"{forecast_values.median():.4f}",
                    f"{forecast_values.std():.4f}",
                    f"{forecast_values.var():.4f}",
                    f"{forecast_values.min():.4f}",
                    f"{forecast_values.max():.4f}",
                    f"{forecast_values.max() - forecast_values.min():.4f}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Trend analysis
            st.write("**Análisis de Tendencia:**")
            
            if len(forecast_values) > 1:
                # Calculate trend
                x = np.arange(len(forecast_values))
                z = np.polyfit(x, forecast_values, 1)
                trend_slope = z[0]
                
                if abs(trend_slope) < 0.01:
                    trend_description = "Estable (sin tendencia clara)"
                elif trend_slope > 0:
                    trend_description = f"Creciente (pendiente: {trend_slope:.4f})"
                else:
                    trend_description = f"Decreciente (pendiente: {trend_slope:.4f})"
                
                st.info(f"**Tendencia:** {trend_description}")
                
                # Volatility analysis
                returns = forecast_values.pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    st.info(f"**Volatilidad Anualizada:** {volatility:.4f}")

with tab4:
    st.subheader("Exportar Análisis Completo en PDF")
    
    st.markdown("""
    **Exportación PDF** genera un reporte completo con todos los resultados del análisis de series temporales.
    
    El reporte incluye:
    - Descripción de los datos
    - Resultados del análisis
    - Información del modelo
    - Pronósticos y estadísticas
    - Gráficos y visualizaciones
    """)
    
    # Check what data is available for the report
    data_available = 'data' in st.session_state and st.session_state.data is not None
    models_available = len(available_models) > 0
    forecasts_available = 'forecast_results' in st.session_state and st.session_state.forecast_results
    
    # Status summary
    st.write("**Estado del Análisis:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data_available:
            st.success("✅ Datos cargados")
        else:
            st.error("❌ Sin datos")
    
    with col2:
        if models_available:
            st.success(f"✅ {len(available_models)} modelo(s) disponible(s)")
        else:
            st.error("❌ Sin modelos")
    
    with col3:
        if forecasts_available:
            st.success("✅ Pronósticos generados")
        else:
            st.warning("⚠️ Sin pronósticos")
    
    # Generate PDF report
    if st.button("📄 Generar Reporte PDF Completo", disabled=not (data_available and models_available)):
        with st.spinner("Generando reporte PDF... Esto puede tomar unos momentos."):
            try:
                # Collect all analysis results
                analysis_results = {}
                
                # Add stationarity results if available
                if 'stationarity_result' in st.session_state:
                    analysis_results['stationarity'] = st.session_state.stationarity_result
                
                # Add decomposition results if available
                if 'decomposition_result' in st.session_state:
                    analysis_results['decomposition'] = st.session_state.decomposition_result
                
                # Collect model results
                model_results = available_models
                
                # Collect forecast results
                forecast_results = None
                if forecasts_available:
                    forecast_results = st.session_state.forecast_results
                
                # Generate PDF
                pdf_data, filename = create_downloadable_pdf(
                    data=st.session_state.data,
                    analysis_results=analysis_results,
                    model_results=model_results,
                    forecast_results=forecast_results
                )
                
                if pdf_data:
                    st.success("¡Reporte PDF generado exitosamente!")
                    
                    # Provide download button
                    st.download_button(
                        label="📥 Descargar Reporte PDF",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        help="Haz clic para descargar el reporte completo en PDF"
                    )
                    
                    # Show report summary
                    st.info(f"""
                    **Reporte Generado:**
                    - Archivo: {filename}
                    - Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                    - Tamaño: {len(pdf_data) / 1024:.1f} KB
                    """)
                else:
                    st.error("Error generando el reporte PDF.")
            
            except Exception as e:
                st.error(f"Error generando reporte: {str(e)}")
    
    # Export forecast data as CSV
    if forecasts_available:
        st.markdown("---")
        st.write("**Exportación de Datos de Pronóstico:**")
        
        if st.button("📊 Exportar Pronósticos como CSV"):
            try:
                forecast_data = st.session_state.forecast_results['forecasts']
                csv_data = forecast_data.to_csv(index=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"pronosticos_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Descargar CSV de Pronósticos",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv"
                )
                
                st.success("Datos de pronóstico listos para descargar!")
            
            except Exception as e:
                st.error(f"Error exportando CSV: {str(e)}")