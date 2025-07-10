import streamlit as st
import pandas as pd
import numpy as np
from utils.data_import import initialize_session_state
from utils.auth import init_auth_db, check_authentication, show_login_form, show_register_form, show_auth_sidebar

# Configure the page
st.set_page_config(
    page_title="TIME FLOW - Time Series Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and session state
init_auth_db()
initialize_session_state()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .auth-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Show authentication status in sidebar
show_auth_sidebar()

# Check if user is authenticated
if not check_authentication():
    # Show login/register interface
    st.markdown("""
    <div class="main-header">
        <h1>📊 TIME FLOW</h1>
        <p>Comprehensive Time Series Analysis & Forecasting Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login/Register tabs
    tab1, tab2 = st.tabs(["🔐 Iniciar Sesión", "📝 Registro"])
    
    with tab1:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        show_login_form()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        show_register_form()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# Main application for authenticated users
st.markdown("""
<div class="main-header">
    <h1>📊 TIME FLOW</h1>
    <p>Comprehensive Time Series Analysis & Forecasting Platform</p>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
## 🚀 Bienvenido a TIME FLOW

**TIME FLOW** es una aplicación completa de análisis de series temporales que te permite realizar análisis end-to-end de tus datos temporales, desde la importación hasta la predicción.

### 📋 Flujo de Trabajo Recomendado:

1. **📥 Importación de Datos** - Carga tus datos o genera datos de ejemplo
2. **🔧 Preprocesamiento** - Limpia y prepara tus datos para el análisis
3. **📊 Visualización** - Explora tus datos con gráficos interactivos
4. **🔍 Análisis** - Realiza análisis de series temporales y pruebas de estacionariedad
5. **🎯 Modelado** - Construye y valida modelos AR, ARIMA, GARCH, EGARCH
6. **🔮 Pronóstico** - Genera predicciones y guarda análisis en PDF

### 🛠️ Características Principales:
""")

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Análisis Completo</h4>
        <ul>
            <li>Importación múltiple de datos</li>
            <li>Preprocesamiento automático</li>
            <li>Visualizaciones interactivas</li>
            <li>Análisis estadístico avanzado</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Modelado Avanzado</h4>
        <ul>
            <li>Modelos AR, ARIMA, GARCH, EGARCH</li>
            <li>Validación de modelos</li>
            <li>Selección automática de parámetros</li>
            <li>Diagnóstico de residuos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🔮 Pronóstico</h4>
        <ul>
            <li>Predicciones con intervalos de confianza</li>
            <li>Análisis de escenarios</li>
            <li>Validación cruzada</li>
            <li>Exportación PDF completa</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Current data status
if 'data' in st.session_state and st.session_state.data is not None:
    st.success(f"✅ Datos cargados: {len(st.session_state.data)} observaciones")
    
    # Show basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", len(st.session_state.data))
    with col2:
        st.metric("Columnas", len(st.session_state.data.columns))
    with col3:
        st.metric("Rango de Fechas", f"{st.session_state.data.index.min().strftime('%Y-%m-%d')} a {st.session_state.data.index.max().strftime('%Y-%m-%d')}")
    with col4:
        st.metric("Valores Faltantes", st.session_state.data.isnull().sum().sum())
    
    # Show data preview
    st.subheader("Vista Previa de Datos")
    st.dataframe(st.session_state.data.head(10))
    
else:
    st.info("👈 Por favor comienza importando tus datos usando la página de **Importación de Datos** en la barra lateral.")

# Navigation instructions
st.markdown("""
## 🎯 Cómo Empezar

1. **Usa la barra lateral** para navegar entre las diferentes páginas
2. **Comienza con la importación** de datos o genera datos de ejemplo
3. **Sigue el flujo secuencial** para obtener mejores resultados
4. **Explora las visualizaciones** interactivas en cada paso

## 📊 Datos de Ejemplo

Si es tu primera vez usando TIME FLOW, te recomendamos:

1. Ir a la página de **Importación de Datos**
2. Usar la función **"Generar Datos Simulados"**
3. Seleccionar el patrón **"Tendencia con Estacionalidad"**
4. Continuar con el flujo de trabajo

---

**¡Comienza tu análisis de series temporales ahora!** 🚀
""")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>TIME FLOW - Análisis de Series Temporales © 2024</p>
</div>
""", unsafe_allow_html=True)
