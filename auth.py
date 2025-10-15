import inspect
import streamlit as st
import bcrypt

# Reemplaza por tus hashes (pueden ser str o bytes)
USERS = {
    "admin": {"name": "Ilya", "hash": "$2b$12$1DiUTnV5R/.qd.qVXkwcl.E/8aBpPLX2tCE3YMTYx00pDZG9EUUYq"},
    "juan":  {"name": "Juan", "hash": "$2b$12$SxF2bjFuPXsHlfadHgEz1.7MmkOd6cqPSXEO5iv3hWCCEKgXkzzwC"},
}

def _resolve_user(user_input: str):
    if not user_input:
        return None, None
    ui = user_input.strip().lower()
    for uname, u in USERS.items():
        if ui == uname.lower() or ui == u["name"].strip().lower():
            return uname, u
    return None, None

def _hash_to_bytes(h):
    return h if isinstance(h, (bytes, bytearray)) else str(h).encode("utf-8")

def require_login() -> bool:
    if st.session_state.get("auth_user"):
        return True

    st.header("Iniciar sesión")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Usuario (nombre o alias)")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar")

    if submitted:
        uname, u = _resolve_user(username)
        if not u:
            st.error("Usuario no encontrado.")
            return False
        ok = bcrypt.checkpw(password.encode("utf-8"), _hash_to_bytes(u["hash"]))
        if ok:
            st.session_state.auth_user = {"username": uname, "name": u["name"]}
            st.success(f"Bienvenido, {u['name']}")
            try: st.rerun()
            except Exception: pass
            return True
        st.error("Contraseña incorrecta.")
        return False

    st.info("Introduce tus credenciales.")
    return False

def logout_button():
    if st.button("Cerrar sesión", key="btn_logout"):
        for k in ["auth_user", "view", "raw", "df_active"]:
            st.session_state.pop(k, None)
        try: st.rerun()
        except Exception: pass