import inspect
import streamlit as st
try:
    import streamlit_authenticator as stauth
except ImportError:
    st.error("Falta 'streamlit-authenticator'. Instala: pip install streamlit-authenticator bcrypt")
    st.stop()

# Pega aquí tus HASHES (no contraseñas en claro)
CREDENTIALS = {
    "usernames": {
        "admin": {"name": "Ilya", "password": "$2b$12$HASH_ADMIN_AQUI"},
        "juan":  {"name": "Juan", "password": "$2b$12$HASH_JUAN_AQUI"},
    }
}

COOKIE_NAME = "ff_auth"
COOKIE_KEY = "ff_auth_key"  # cambia por un token aleatorio
COOKIE_DAYS = 7

def _login_compat(authenticator):
    sig = inspect.signature(authenticator.login)
    params = sig.parameters
    # API nueva: location como keyword (y opcional fields)
    if "location" in params:
        if "fields" in params:
            return authenticator.login(
                location="main",
                fields={"Form name": "Iniciar sesión", "Username": "Usuario", "Password": "Contraseña"},
            )
        return authenticator.login(location="main")
    # API antigua: (form_name, location)
    return authenticator.login("Iniciar sesión", "main")

def _logout_compat(authenticator):
    try:
        authenticator.logout(location="sidebar")
    except TypeError:
        authenticator.logout("Cerrar sesión", "sidebar")

def require_login() -> bool:
    authenticator = stauth.Authenticate(CREDENTIALS, COOKIE_NAME, COOKIE_KEY, COOKIE_DAYS)
    name, status, username = _login_compat(authenticator)

    if status is True:
        st.session_state.user = {"name": name, "username": username}
        with st.sidebar:
            _logout_compat(authenticator)
        return True
    elif status is False:
        st.error("Usuario o contraseña incorrectos.")
        return False
    else:
        st.info("Introduce tus credenciales.")
        return False