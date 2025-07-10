import streamlit as st
import hashlib
import sqlite3
import os
import re
from datetime import datetime

# Database setup for authentication
def init_auth_db():
    """Initialize authentication database"""
    conn = sqlite3.connect('auth.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "La contraseña debe tener al menos 6 caracteres"
    return True, ""

def register_user(name, email, password):
    """Register a new user"""
    try:
        # Validate inputs
        if not name.strip():
            return False, "El nombre es requerido"
        
        if not validate_email(email):
            return False, "Formato de correo inválido"
        
        valid_password, password_error = validate_password(password)
        if not valid_password:
            return False, password_error
        
        conn = sqlite3.connect('auth.db')
        cursor = conn.cursor()
        
        # Check if email already exists
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "El correo ya está registrado"
        
        # Insert new user
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name.strip(), email.lower(), password_hash)
        )
        
        conn.commit()
        conn.close()
        return True, "Usuario registrado exitosamente"
        
    except Exception as e:
        return False, f"Error al registrar usuario: {str(e)}"

def authenticate_user(email, password):
    """Authenticate user login"""
    try:
        conn = sqlite3.connect('auth.db')
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute(
            "SELECT id, name, email FROM users WHERE email = ? AND password_hash = ?",
            (email.lower(), password_hash)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return True, {
                'id': user[0],
                'name': user[1],
                'email': user[2]
            }
        else:
            return False, "Correo o contraseña incorrectos"
            
    except Exception as e:
        return False, f"Error al autenticar: {str(e)}"

def check_authentication():
    """Check if user is authenticated"""
    return 'authenticated' in st.session_state and st.session_state.authenticated

def login_required():
    """Decorator function to require login for pages"""
    if not check_authentication():
        st.error("Debe iniciar sesión para acceder a esta página")
        st.stop()

def logout_user():
    """Logout current user"""
    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']
    if 'user_info' in st.session_state:
        del st.session_state['user_info']

def get_current_user():
    """Get current authenticated user info"""
    if check_authentication():
        return st.session_state.get('user_info', {})
    return None

def show_login_form():
    """Display login form"""
    st.subheader("🔐 Iniciar Sesión")
    
    with st.form("login_form"):
        email = st.text_input("Correo Electrónico")
        password = st.text_input("Contraseña", type="password")
        submit_button = st.form_submit_button("Iniciar Sesión")
        
        if submit_button:
            if email and password:
                success, result = authenticate_user(email, password)
                
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_info = result
                    st.success(f"Bienvenido, {result['name']}!")
                    st.rerun()
                else:
                    st.error(result)
            else:
                st.error("Por favor complete todos los campos")

def show_register_form():
    """Display registration form"""
    st.subheader("📝 Registro")
    
    with st.form("register_form"):
        name = st.text_input("Nombre Completo")
        email = st.text_input("Correo Electrónico")
        password = st.text_input("Contraseña", type="password")
        confirm_password = st.text_input("Confirmar Contraseña", type="password")
        submit_button = st.form_submit_button("Registrarse")
        
        if submit_button:
            if name and email and password and confirm_password:
                if password != confirm_password:
                    st.error("Las contraseñas no coinciden")
                else:
                    success, message = register_user(name, email, password)
                    
                    if success:
                        st.success(message)
                        st.info("Ahora puede iniciar sesión con sus credenciales")
                    else:
                        st.error(message)
            else:
                st.error("Por favor complete todos los campos")

def show_auth_sidebar():
    """Show authentication status in sidebar"""
    if check_authentication():
        user_info = get_current_user()
        st.sidebar.success(f"👤 {user_info['name']}")
        if st.sidebar.button("Cerrar Sesión"):
            logout_user()
            st.rerun()
    else:
        st.sidebar.warning("No autenticado")