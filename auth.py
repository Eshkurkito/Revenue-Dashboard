import inspect
import streamlit as st
try:
    import streamlit_authenticator as stauth
except ImportError:
    st.error("Falta 'streamlit-authenticator'. Instala: pip install streamlit-authenticator bcrypt")
    st.stop()

# Pega aquí los HASHES reales (empiezan por $2b$...), no contraseñas en claro
CREDENTIALS = {
    "usernames": {
        "admin": {"name": "Ilya", "password": "$2b$12$uPVHJLuX2eNnd6VAfx.L3ugH09tq9tdID2eTA4TUY3qkqaYGTZF4."},
        "juan":  {"name": "Juan", "password": "$2b$12$YVBmi65h6ABHZYKWyLDdEeLYKHUAIaA.D8xoQLFwmTgGqy42/SrJ."},
    }
}

COOKIE_NAME = "ff_auth"
COOKIE_KEY = "ff_auth_key"  # cambia por un token aleatorio
COOKIE_DAYS = 7

def _login_compat(authenticator):
    # Prueba las firmas nuevas y antiguas y normaliza a 3 valores
    try:
        res = authenticator.login(
            location="main",
            fields={"Form name": "Iniciar sesión", "Username": "Usuario", "Password": "Contraseña"},
        )
    except TypeError:
        try:
            res = authenticator.login(location="main")
        except TypeError:
            res = authenticator.login("Iniciar sesión", "main")

    if not isinstance(res, tuple):
        return None, None, None
    if len(res) == 3:
        return res
    if len(res) == 2:
        name, status = res
        return name, status, (name or "")
    return None, None, None

def _logout_compat(authenticator):
    try:
        authenticator.logout(location="sidebar")
    except TypeError:
        try:
            authenticator.logout("Cerrar sesión", "sidebar")
        except Exception:
            pass

def require_login() -> bool:
    authenticator = stauth.Authenticate(CREDENTIALS, COOKIE_NAME, COOKIE_KEY, COOKIE_DAYS)
    name, status, username = _login_compat(authenticator)

    if status is True:
        st.session_state.user = {"name": name or username, "username": username or name}
        with st.sidebar:
            _logout_compat(authenticator)
        return True
    if status is False:
        st.error("Usuario o contraseña incorrectos.")
        return False
    st.info("Introduce tus credenciales.")
    return False