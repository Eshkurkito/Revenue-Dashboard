import streamlit as st
try:
    import streamlit_authenticator as stauth
except ImportError:
    st.error("Falta 'streamlit-authenticator'. Instala: pip install streamlit-authenticator bcrypt")
    st.stop()

# Sustituye los hashes por los tuyos
CREDENTIALS = {
    "usernames": {
        "admin": {"name": "Ilya", "password": "Ilya2025"},
        "juan":  {"name": "Juan",  "password": "Juan2025"},
    }
}

COOKIE_NAME = "ff_auth"
COOKIE_KEY = "ff_auth_key"  # cambia por uno propio/aleatorio
COOKIE_DAYS = 7

def require_login() -> bool:
    authenticator = stauth.Authenticate(CREDENTIALS, COOKIE_NAME, COOKIE_KEY, COOKIE_DAYS)

    # API nueva (>=0.3): usa keywords location y fields
    try:
        name, status, username = authenticator.login(
            location="main",
            fields={
                "Form name": "Iniciar sesión",
                "Username": "Usuario",
                "Password": "Contraseña",
            },
        )
    except TypeError:
        # Compatibilidad API antigua: (form_name, location)
        name, status, username = authenticator.login("Iniciar sesión", "main")

    if status is True:
        st.session_state.user = {"name": name, "username": username}
        with st.sidebar:
            # API nueva: location como keyword
            try:
                authenticator.logout(location="sidebar")
            except TypeError:
                authenticator.logout("Cerrar sesión", "sidebar")
        return True
    elif status is False:
        st.error("Usuario o contraseña incorrectos.")
        return False
    else:
        st.info("Introduce tus credenciales.")
        return False