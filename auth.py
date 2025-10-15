import streamlit as st
import streamlit_authenticator as stauth

# Sustituye <HASH1>, <HASH2> por hashes reales
CREDENTIALS = {
    "usernames": {
        "admin": {"name": "Admin", "password": "Florit2025!"},
        "juan":  {"name": "Marta",  "password": "Florit2025"},
    }
}
COOKIE_NAME = "ff_auth"
COOKIE_KEY = "ff_auth_key"
COOKIE_DAYS = 7

def require_login() -> bool:
    authenticator = stauth.Authenticate(CREDENTIALS, COOKIE_NAME, COOKIE_KEY, COOKIE_DAYS)
    name, status, username = authenticator.login("Iniciar sesión", "main")
    if status:
        st.session_state.user = {"name": name, "username": username}
        with st.sidebar:
            authenticator.logout("Cerrar sesión")
        return True
    if status is False:
        st.error("Usuario o contraseña incorrectos.")
    return False