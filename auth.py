import inspect
import streamlit as st
import bcrypt

# Pega aquí los hashes generados con generate_hashes.py
USERS = {
    "admin": {"name": "Ilya", "hash": b"$2b$12$uPVHJLuX2eNnd6VAfx.L3ugH09tq9tdID2eTA4TUY3qkqaYGTZF4"},
    "juan":  {"name": "Juan", "hash": b"$2b$12$YVBmi65h6ABHZYKWyLDdEeLYKHUAIaA.D8xoQLFwmTgGqy42/SrJ"},
}

def _resolve_user(user_input: str):
    if not user_input:
        return None, None
    ui = user_input.strip().lower()
    for uname, u in USERS.items():
        # Acepta tanto el "username" (clave) como el "name" mostrado
        if ui == uname.lower() or ui == u["name"].strip().lower():
            return uname, u
    return None, None

def require_login() -> bool:
    # Ya logado
    if st.session_state.get("auth_user"):
        return True

    st.header("Iniciar sesión")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Usuario (nombre o alias)")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar")

    if submitted:
        uname, u = _resolve_user(username)
        if u and bcrypt.checkpw(password.encode("utf-8"), u["hash"]):
            st.session_state.auth_user = {"username": uname, "name": u["name"]}
            st.success(f"Bienvenido, {u['name']}")
            try:
                st.rerun()
            except Exception:
                pass
            return True
        st.error("Usuario o contraseña incorrectos.")
        return False

    st.info("Introduce tus credenciales.")
    return False

def logout_button():
    if st.button("Cerrar sesión", key="btn_logout"):
        for k in ["auth_user", "view", "raw", "df_active"]:
            st.session_state.pop(k, None)
        try:
            st.rerun()
        except Exception:
            pass