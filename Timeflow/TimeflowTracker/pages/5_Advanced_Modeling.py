import streamlit as st
import pandas as pd
import numpy as np
from utils.advanced_models import (
    fit_ar_model, fit_arima_model, fit_garch_model, fit_egarch_model,
    validate_model, compare_models as compare_advanced_models,
    generate_model_forecasts, auto_arima_selection
)
from utils.visualization import create_time_series_plot
from utils.data_import import initialize_session_state
from utils.auth import login_required, show_auth_sidebar
import plotly.graph_objects as go
import plotly.express as px

# Initialize session state
initialize_session_state()

# Check authentication
login_required()

# Show authentication status in sidebar
show_auth_sidebar()

st.title("🎯 Modelado Avanzado")
st.markdown("Construye y valida modelos AR, ARIMA, GARCH y EGARCH para tus datos de series temporales.")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ No hay datos disponibles. Por favor carga datos primero usando la página de **Importación de Datos**.")
    st.stop()

data = st.session_state.data

# Initialize model results in session state
if 'advanced_models' not in st.session_state:
    st.session_state.advanced_models = {}

# Model selection
st.subheader("📊 Selección de Modelo")

model_descriptions = {
    "AR": "**Modelo AutoRegresivo (AR)**: Predice valores futuros basándose en valores pasados de la serie temporal.",
    "ARIMA": "**Modelo ARIMA**: Combina autoregresión, integración y medias móviles para series no estacionarias.",
    "GARCH": "**Modelo GARCH**: Modela la volatilidad cambiante en el tiempo, útil para datos financieros.",
    "EGARCH": "**Modelo EGARCH**: Modelo GARCH asimétrico que captura efectos de leverage en la volatilidad."
}

for model_type, description in model_descriptions.items():
    st.markdown(description)

st.markdown("---")

# Create tabs for different models
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Modelo AR", 
    "📊 Modelo ARIMA", 
    "📉 Modelo GARCH",
    "🔄 Modelo EGARCH",
    "📋 Comparación"
])

with tab1:
    st.subheader("Modelo AutoRegresivo (AR)")
    
    st.markdown("""
    **El modelo AR(p)** predice el valor actual basándose en p valores pasados de la serie temporal.
    
    **Ecuación:** X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t
    """)
    
    # AR model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        ar_lags = st.number_input(
            "Número de Lags (p)",
            min_value=1,
            max_value=20,
            value=2,
            help="Número de valores pasados a incluir en el modelo"
        )
    
    with col2:
        ar_auto_select = st.checkbox(
            "Selección automática de lags",
            value=True,
            help="Usar AIC para seleccionar el número óptimo de lags"
        )
    
    # Fit AR model
    if st.button("Ajustar Modelo AR"):
        with st.spinner("Ajustando modelo AR..."):
            ar_lags_final = None if ar_auto_select else ar_lags
            
            ar_result, error = fit_ar_model(data.iloc[:, 0], lags=ar_lags_final)
            
            if error:
                st.error(f"Error ajustando modelo AR: {error}")
            else:
                st.session_state.advanced_models['AR'] = ar_result
                st.success("¡Modelo AR ajustado exitosamente!")
                
                # Show model results
                st.subheader("📊 Resultados del Modelo AR")
                
                # Model equation
                st.markdown("### 📐 Ecuación del Modelo")
                lags = ar_result['lags']
                equation = f"X_t = c"
                for i in range(1, lags + 1):
                    equation += f" + φ_{i}X_{{t-{i}}}"
                equation += " + ε_t"
                st.latex(equation)
                
                # Model parameters
                st.markdown("### 📊 Parámetros del Modelo")
                params_df = pd.DataFrame({
                    'Parámetro': ar_result['params'].index,
                    'Valor': ar_result['params'].values,
                    'Descripción': ['Constante'] + [f'Coeficiente AR({i})' for i in range(1, lags + 1)]
                })
                st.dataframe(params_df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lags Seleccionados", ar_result['lags'])
                with col2:
                    st.metric("AIC", f"{ar_result['aic']:.2f}")
                with col3:
                    st.metric("BIC", f"{ar_result['bic']:.2f}")
                
                # Model analysis and recommendations
                st.markdown("### 🎯 Análisis y Recomendaciones")
                
                # Check parameter significance (simplified)
                significant_params = sum(1 for p in ar_result['params'][1:] if abs(p) > 0.1)
                total_params = len(ar_result['params']) - 1
                
                recommendations = []
                
                if ar_result['aic'] < 100:
                    recommendations.append("✅ El modelo muestra un buen ajuste según el criterio AIC.")
                else:
                    recommendations.append("⚠️ El AIC es alto, considera revisar el número de lags.")
                
                if significant_params == total_params:
                    recommendations.append("✅ Todos los parámetros AR parecen ser significativos.")
                elif significant_params < total_params / 2:
                    recommendations.append("⚠️ Algunos parámetros AR pueden no ser significativos. Considera reducir el número de lags.")
                
                recommendations.append(f"📊 **Uso recomendado:** Este modelo AR({lags}) es adecuado para series con dependencia temporal a corto plazo.")
                recommendations.append("📈 **Para pronósticos:** Utiliza este modelo para horizontes de pronóstico cortos (1-5 períodos).")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Plot fitted vs actual
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.iloc[:, 0],
                    mode='lines',
                    name='Datos Originales',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=ar_result['fitted_values'].index,
                    y=ar_result['fitted_values'],
                    mode='lines',
                    name='Valores Ajustados',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Modelo AR - Datos vs Valores Ajustados',
                    xaxis_title='Fecha',
                    yaxis_title='Valor',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model validation
                validation_result, val_error = validate_model(ar_result, data.iloc[:, 0])
                
                if not val_error:
                    st.subheader("✅ Validación del Modelo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RMSE", f"{validation_result['rmse']:.4f}")
                    with col2:
                        st.metric("MAE", f"{validation_result['mae']:.4f}")
                    with col3:
                        st.metric("MAPE", f"{validation_result['mape']:.2f}%")
                    with col4:
                        st.metric("Residuos μ", f"{validation_result['residuals_mean']:.4f}")

with tab2:
    st.subheader("Modelo ARIMA")
    
    st.markdown("""
    **El modelo ARIMA(p,d,q)** combina autoregresión, integración y medias móviles.
    
    - **p**: Orden autoregresivo
    - **d**: Grado de diferenciación
    - **q**: Orden de media móvil
    """)
    
    # ARIMA model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        arima_manual = st.checkbox("Configuración Manual", value=False)
    
    with col2:
        arima_auto = st.checkbox("Selección Automática", value=True)
    
    if arima_manual:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
        with col2:
            d = st.number_input("d (Differencing)", min_value=0, max_value=3, value=1)
        with col3:
            q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
        
        arima_order = (p, d, q)
    else:
        arima_order = (1, 1, 1)  # Default
    
    # Fit ARIMA model
    if st.button("Ajustar Modelo ARIMA"):
        with st.spinner("Ajustando modelo ARIMA..."):
            if arima_auto:
                arima_result, error = auto_arima_selection(data.iloc[:, 0])
            else:
                arima_result, error = fit_arima_model(data.iloc[:, 0], order=arima_order)
            
            if error:
                st.error(f"Error ajustando modelo ARIMA: {error}")
            else:
                st.session_state.advanced_models['ARIMA'] = arima_result
                st.success("¡Modelo ARIMA ajustado exitosamente!")
                
                # Show model results
                st.subheader("📊 Resultados del Modelo ARIMA")
                
                # Model equation
                st.markdown("### 📐 Ecuación del Modelo")
                p, d, q = arima_result['order']
                
                if d == 0:
                    st.latex(r"X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t")
                elif d == 1:
                    st.latex(r"\Delta X_t = c + \sum_{i=1}^{p} \phi_i \Delta X_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t")
                else:
                    st.latex(f"\\Delta^{{{d}}} X_t = c + \\sum_{{i=1}}^{{{p}}} \\phi_i \\Delta^{{{d}}} X_{{t-i}} + \\sum_{{j=1}}^{{{q}}} \\theta_j \\epsilon_{{t-j}} + \\epsilon_t")
                
                st.markdown(f"""
                **Donde:**
                - p = {p} (términos autoregresivos)
                - d = {d} (diferenciaciones)
                - q = {q} (términos de media móvil)
                """)
                
                # Model parameters
                st.markdown("### 📊 Parámetros del Modelo")
                params_df = pd.DataFrame({
                    'Parámetro': arima_result['params'].index,
                    'Valor': arima_result['params'].values
                })
                st.dataframe(params_df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Orden", str(arima_result['order']))
                with col2:
                    st.metric("AIC", f"{arima_result['aic']:.2f}")
                with col3:
                    st.metric("BIC", f"{arima_result['bic']:.2f}")
                
                # Model analysis and recommendations
                st.markdown("### 🎯 Análisis y Recomendaciones")
                
                recommendations = []
                
                if arima_result['aic'] < 50:
                    recommendations.append("✅ Excelente ajuste del modelo según el criterio AIC.")
                elif arima_result['aic'] < 100:
                    recommendations.append("✅ Buen ajuste del modelo según el criterio AIC.")
                else:
                    recommendations.append("⚠️ El AIC es alto, considera revisar los parámetros del modelo.")
                
                if d == 0:
                    recommendations.append("📊 La serie es estacionaria (d=0), no requiere diferenciación.")
                elif d == 1:
                    recommendations.append("📊 Se aplicó una diferenciación para lograr estacionariedad.")
                else:
                    recommendations.append(f"⚠️ Se aplicaron {d} diferenciaciones. Verifica si es necesario.")
                
                if p > 0 and q > 0:
                    recommendations.append("📈 Modelo mixto AR-MA captura tanto dependencia autoregresiva como de errores.")
                elif p > 0:
                    recommendations.append("📈 Modelo predominantemente autoregresivo (AR).")
                elif q > 0:
                    recommendations.append("📈 Modelo predominantemente de medias móviles (MA).")
                
                recommendations.append(f"🎯 **Uso recomendado:** ARIMA({p},{d},{q}) es adecuado para series no estacionarias con patrones complejos.")
                recommendations.append("📊 **Para pronósticos:** Útil para horizontes de pronóstico medianos a largos (5-20 períodos).")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Plot fitted vs actual
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.iloc[:, 0],
                    mode='lines',
                    name='Datos Originales',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=arima_result['fitted_values'].index,
                    y=arima_result['fitted_values'],
                    mode='lines',
                    name='Valores Ajustados',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Modelo ARIMA - Datos vs Valores Ajustados',
                    xaxis_title='Fecha',
                    yaxis_title='Valor',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model validation
                validation_result, val_error = validate_model(arima_result, data.iloc[:, 0])
                
                if not val_error:
                    st.subheader("✅ Validación del Modelo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RMSE", f"{validation_result['rmse']:.4f}")
                    with col2:
                        st.metric("MAE", f"{validation_result['mae']:.4f}")
                    with col3:
                        st.metric("MAPE", f"{validation_result['mape']:.2f}%")
                    with col4:
                        st.metric("Residuos μ", f"{validation_result['residuals_mean']:.4f}")

with tab3:
    st.subheader("Modelo GARCH")
    
    st.markdown("""
    **El modelo GARCH(p,q)** modela la volatilidad heterocedástica condicional.
    
    Es especialmente útil para datos financieros donde la volatilidad cambia con el tiempo.
    """)
    
    # GARCH model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        garch_p = st.number_input("p (GARCH order)", min_value=1, max_value=3, value=1, key="garch_p_input")
    
    with col2:
        garch_q = st.number_input("q (ARCH order)", min_value=1, max_value=3, value=1, key="garch_q_input")
    
    # Fit GARCH model
    if st.button("Ajustar Modelo GARCH"):
        with st.spinner("Ajustando modelo GARCH..."):
            garch_result, error = fit_garch_model(data.iloc[:, 0], p=garch_p, q=garch_q)
            
            if error:
                st.error(f"Error ajustando modelo GARCH: {error}")
            else:
                st.session_state.advanced_models['GARCH'] = garch_result
                st.success("¡Modelo GARCH ajustado exitosamente!")
                
                # Show model results
                st.subheader("📊 Resultados del Modelo GARCH")
                
                # Model equation
                st.markdown("### 📐 Ecuación del Modelo")
                p, q = garch_result['p'], garch_result['q']
                
                st.latex(r"r_t = \mu + \epsilon_t")
                st.latex(r"\epsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)")
                st.latex(f"\\sigma_t^2 = \\omega + \\sum_{{i=1}}^{{{q}}} \\alpha_i \\epsilon_{{t-i}}^2 + \\sum_{{j=1}}^{{{p}}} \\beta_j \\sigma_{{t-j}}^2")
                
                st.markdown(f"""
                **Donde:**
                - r_t = retorno en el tiempo t
                - σ_t² = varianza condicional (volatilidad)
                - ω = término constante
                - α_i = coeficientes ARCH (q = {q})
                - β_j = coeficientes GARCH (p = {p})
                - ε_t = término de error
                """)
                
                # Model parameters
                st.markdown("### 📊 Parámetros del Modelo")
                params_df = pd.DataFrame({
                    'Parámetro': garch_result['params'].index,
                    'Valor': garch_result['params'].values
                })
                st.dataframe(params_df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Orden GARCH", f"({garch_result['p']}, {garch_result['q']})")
                with col2:
                    st.metric("AIC", f"{garch_result['aic']:.2f}")
                with col3:
                    st.metric("BIC", f"{garch_result['bic']:.2f}")
                
                # Model analysis and recommendations
                st.markdown("### 🎯 Análisis y Recomendaciones")
                
                recommendations = []
                
                # Analyze volatility persistence
                vol_mean = garch_result['conditional_volatility'].mean()
                vol_max = garch_result['conditional_volatility'].max()
                vol_min = garch_result['conditional_volatility'].min()
                
                if vol_max / vol_min > 3:
                    recommendations.append("📊 Alta variabilidad en la volatilidad detectada - GARCH es apropiado.")
                else:
                    recommendations.append("⚠️ Baja variabilidad en volatilidad - considera si GARCH es necesario.")
                
                if garch_result['aic'] < 0:
                    recommendations.append("✅ Excelente ajuste del modelo GARCH.")
                elif garch_result['aic'] < 5:
                    recommendations.append("✅ Buen ajuste del modelo GARCH.")
                else:
                    recommendations.append("⚠️ Considera revisar los parámetros o usar un modelo más simple.")
                
                recommendations.append("💰 **Uso recomendado:** Ideal para modelar volatilidad en datos financieros.")
                recommendations.append("⚡ **Clusters de volatilidad:** Este modelo captura períodos de alta y baja volatilidad.")
                recommendations.append("📈 **Para pronósticos:** Excelente para pronósticos de volatilidad a corto-mediano plazo.")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Plot volatility
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=garch_result['returns'].index,
                    y=garch_result['returns'],
                    mode='lines',
                    name='Retornos',
                    line=dict(color='blue', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=garch_result['conditional_volatility'].index,
                    y=garch_result['conditional_volatility'],
                    mode='lines',
                    name='Volatilidad Condicional',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Modelo GARCH - Retornos y Volatilidad Condicional',
                    xaxis_title='Fecha',
                    yaxis_title='Retornos',
                    yaxis2=dict(title='Volatilidad', overlaying='y', side='right'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model validation
                validation_result, val_error = validate_model(garch_result, data.iloc[:, 0])
                
                if not val_error:
                    st.subheader("✅ Validación del Modelo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Log-Likelihood", f"{validation_result['log_likelihood']:.2f}")
                    with col2:
                        st.metric("AIC", f"{validation_result['aic']:.2f}")
                    with col3:
                        st.metric("BIC", f"{validation_result['bic']:.2f}")
                    with col4:
                        st.metric("Vol. Media", f"{validation_result['volatility_mean']:.4f}")

with tab4:
    st.subheader("Modelo EGARCH")
    
    st.markdown("""
    **El modelo EGARCH(p,o,q)** es una extensión asimétrica del modelo GARCH.
    
    Captura el efecto de leverage donde las noticias negativas tienden a aumentar más la volatilidad que las positivas.
    """)
    
    # EGARCH model parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        egarch_p = st.number_input("p (EGARCH order)", min_value=1, max_value=3, value=1, key="egarch_p_input")
    
    with col2:
        egarch_o = st.number_input("o (Asimetría)", min_value=1, max_value=3, value=1, key="egarch_o_input")
    
    with col3:
        egarch_q = st.number_input("q (ARCH order)", min_value=1, max_value=3, value=1, key="egarch_q_input")
    
    # Fit EGARCH model
    if st.button("Ajustar Modelo EGARCH"):
        with st.spinner("Ajustando modelo EGARCH..."):
            egarch_result, error = fit_egarch_model(data.iloc[:, 0], p=egarch_p, o=egarch_o, q=egarch_q)
            
            if error:
                st.error(f"Error ajustando modelo EGARCH: {error}")
            else:
                st.session_state.advanced_models['EGARCH'] = egarch_result
                st.success("¡Modelo EGARCH ajustado exitosamente!")
                
                # Show model results
                st.subheader("📊 Resultados del Modelo EGARCH")
                
                # Model equation
                st.markdown("### 📐 Ecuación del Modelo")
                p, o, q = egarch_result['p'], egarch_result['o'], egarch_result['q']
                
                st.latex(r"r_t = \mu + \epsilon_t")
                st.latex(r"\epsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)")
                st.latex(f"\\ln(\\sigma_t^2) = \\omega + \\sum_{{i=1}}^{{{q}}} \\alpha_i \\left|\\frac{{\\epsilon_{{t-i}}}}{{\\sigma_{{t-i}}}}\\right| + \\sum_{{j=1}}^{{{o}}} \\gamma_j \\frac{{\\epsilon_{{t-j}}}}{{\\sigma_{{t-j}}}} + \\sum_{{k=1}}^{{{p}}} \\beta_k \\ln(\\sigma_{{t-k}}^2)")
                
                st.markdown(f"""
                **Donde:**
                - r_t = retorno en el tiempo t
                - ln(σ_t²) = logaritmo de la varianza condicional
                - ω = término constante
                - α_i = coeficientes de magnitud (q = {q})
                - γ_j = coeficientes de asimetría (o = {o})
                - β_k = coeficientes de persistencia (p = {p})
                - **Efecto leverage:** γ < 0 indica que noticias negativas aumentan más la volatilidad
                """)
                
                # Model parameters
                st.markdown("### 📊 Parámetros del Modelo")
                params_df = pd.DataFrame({
                    'Parámetro': egarch_result['params'].index,
                    'Valor': egarch_result['params'].values
                })
                st.dataframe(params_df, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Orden EGARCH", f"({egarch_result['p']}, {egarch_result['o']}, {egarch_result['q']})")
                with col2:
                    st.metric("AIC", f"{egarch_result['aic']:.2f}")
                with col3:
                    st.metric("BIC", f"{egarch_result['bic']:.2f}")
                
                # Model analysis and recommendations
                st.markdown("### 🎯 Análisis y Recomendaciones")
                
                recommendations = []
                
                # Check for leverage effects
                gamma_params = [param for param in egarch_result['params'].index if 'gamma' in param.lower()]
                if gamma_params:
                    gamma_values = [egarch_result['params'][param] for param in gamma_params]
                    avg_gamma = np.mean(gamma_values)
                    
                    if avg_gamma < -0.1:
                        recommendations.append("📉 **Fuerte efecto leverage:** Noticias negativas aumentan significativamente la volatilidad.")
                    elif avg_gamma < 0:
                        recommendations.append("📊 **Efecto leverage moderado:** Ligera asimetría en respuesta a noticias.")
                    else:
                        recommendations.append("📈 **Sin efecto leverage:** Respuesta simétrica a noticias positivas y negativas.")
                
                if egarch_result['aic'] < 0:
                    recommendations.append("✅ Excelente ajuste del modelo EGARCH.")
                elif egarch_result['aic'] < 5:
                    recommendations.append("✅ Buen ajuste del modelo EGARCH.")
                else:
                    recommendations.append("⚠️ Considera revisar los parámetros o comparar con GARCH simple.")
                
                recommendations.append("💰 **Uso recomendado:** Ideal para mercados financieros con efecto leverage.")
                recommendations.append("⚡ **Ventaja clave:** Captura asimetría en la respuesta de volatilidad.")
                recommendations.append("📈 **Para pronósticos:** Superior a GARCH cuando hay efectos asimétricos.")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Plot volatility
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=egarch_result['returns'].index,
                    y=egarch_result['returns'],
                    mode='lines',
                    name='Retornos',
                    line=dict(color='blue', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=egarch_result['conditional_volatility'].index,
                    y=egarch_result['conditional_volatility'],
                    mode='lines',
                    name='Volatilidad Condicional',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Modelo EGARCH - Retornos y Volatilidad Condicional',
                    xaxis_title='Fecha',
                    yaxis_title='Retornos',
                    yaxis2=dict(title='Volatilidad', overlaying='y', side='right'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model validation
                validation_result, val_error = validate_model(egarch_result, data.iloc[:, 0])
                
                if not val_error:
                    st.subheader("✅ Validación del Modelo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Log-Likelihood", f"{validation_result['log_likelihood']:.2f}")
                    with col2:
                        st.metric("AIC", f"{validation_result['aic']:.2f}")
                    with col3:
                        st.metric("BIC", f"{validation_result['bic']:.2f}")
                    with col4:
                        st.metric("Vol. Media", f"{validation_result['volatility_mean']:.4f}")

with tab5:
    st.subheader("Comparación de Modelos")
    
    if len(st.session_state.advanced_models) == 0:
        st.info("No hay modelos ajustados para comparar. Ajusta al menos un modelo en las pestañas anteriores.")
    else:
        st.markdown("### 📊 Tabla de Comparación")
        
        # Create comparison table
        comparison_data = []
        for model_name, model_result in st.session_state.advanced_models.items():
            comparison_data.append({
                'Modelo': model_name,
                'Tipo': model_result['model_type'],
                'AIC': model_result['aic'],
                'BIC': model_result['bic'],
                'Parámetros': len(model_result['params'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by AIC
        comparison_df = comparison_df.sort_values('AIC')
        
        # Display table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model
        best_model = comparison_df.iloc[0]
        st.success(f"🏆 **Mejor Modelo (por AIC):** {best_model['Modelo']} (AIC: {best_model['AIC']:.2f})")
        
        # Model performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=comparison_df['Modelo'],
            y=comparison_df['AIC'],
            name='AIC',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=comparison_df['Modelo'],
            y=comparison_df['BIC'],
            name='BIC',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Comparación de Modelos - Criterios de Información',
            xaxis_title='Modelo',
            yaxis_title='Valor del Criterio',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model recommendations
        st.subheader("🎯 Recomendaciones")
        
        if len(st.session_state.advanced_models) >= 2:
            st.markdown("""
            **Interpretación de los Criterios:**
            
            - **AIC (Akaike Information Criterion)**: Menor es mejor. Balancea bondad de ajuste con complejidad.
            - **BIC (Bayesian Information Criterion)**: Menor es mejor. Penaliza más la complejidad que AIC.
            
            **Recomendación:** Utiliza el modelo con menor AIC para pronósticos. Si hay diferencias pequeñas, 
            prefiere el modelo más simple (menor BIC).
            """)
        
        # Clear models button
        if st.button("🗑️ Limpiar Todos los Modelos"):
            st.session_state.advanced_models = {}
            st.success("Todos los modelos han sido eliminados.")
            st.rerun()