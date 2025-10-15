import pandas as pd
import streamlit as st
from auth import require_login, logout_button
from landing_ui import render_landing, get_logo_path

# --- helpers ---
def cargar_dataframe():
    return pd.DataFrame()

def _get_raw():
    return st.session_state.get("df_active") or st.session_state.get("raw")

def _safe_call(view_fn):
    raw = _get_raw()
    try:
        return view_fn(raw)
    except TypeError:
        return view_fn()
    except (KeyError, ValueError):
        st.info("Necesitas cargar un archivo antes de acceder a este módulo.")
        return

def _rerun():
    try:
        st.rerun()
    except Exception:
        pass

def clear_data(reset_view: bool = True):
    # Limpia dataset y cache
    for k in ["raw", "df_active"]:
        st.session_state.pop(k, None)
    try:
        st.cache_data.clear()
    except Exception:
        pass
    if reset_view:
        st.session_state.view = "landing"
        _rerun()

# --- login ---
if not require_login():
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    logo = get_logo_path()
    if logo:
        st.image(logo, width=140)

    # Botones navegación y logout
    if st.button("⬅️ Volver al inicio", key="btn_home", use_container_width=True):
        st.session_state.view = "landing"; _rerun()
    logout_button()

    st.header("Carga de datos")
    # Preferencia: auto-borrar al quitar archivo del uploader
    st.checkbox("Borrar datos al quitar el archivo", value=True, key="auto_clear_on_remove")

    uploaded_file = st.file_uploader("Carga tu archivo Excel/CSV", type=["xlsx", "csv"], key="uploader")

    # Detecta cuando quitas el archivo del uploader (flanco de 1->0)
    prev_has = st.session_state.get("uploader_has_file", False)
    curr_has = uploaded_file is not None
    st.session_state.uploader_has_file = curr_has
    if prev_has and not curr_has and st.session_state.get("auto_clear_on_remove", True) and _get_raw() is not None:
        clear_data(reset_view=False)
        st.success("Datos borrados al quitar el archivo.")

    # Cargar archivo manualmente
    if uploaded_file is not None and st.button("Usar este archivo", key="btn_use_upload", use_container_width=True):
        try:
            if uploaded_file.name.lower().endswith(".xlsx"):
                st.session_state.raw = pd.read_excel(uploaded_file)
            else:
                st.session_state.raw = pd.read_csv(uploaded_file)
            st.success("Archivo cargado correctamente.")
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")

    # Botón para limpiar datos cargados en cualquier momento
    if _get_raw() is not None:
        if st.button("🧹 Quitar datos cargados", key="btn_clear_data", use_container_width=True):
            clear_data()

# --- Router (sin cambios) ---
VALID_VIEWS = {"landing", "consulta", "pro", "whatif", "evolucion"}

# Login primero
if not require_login():
    st.stop()

# Asegura vista por defecto tras login
if st.session_state.get("view") not in VALID_VIEWS:
    st.session_state.view = "landing"

# --- Router ---
if st.session_state.view == "landing":
    render_landing()
elif st.session_state.view == "consulta":
    from consulta_normal import render_consulta_normal
    _safe_call(render_consulta_normal)
elif st.session_state.view == "pro":
    from cuadro_mando_pro import render_cuadro_mando_pro
    _safe_call(render_cuadro_mando_pro)
elif st.session_state.view == "whatif":
    from what_if import render_what_if
    _safe_call(render_what_if)
elif st.session_state.view == "evolucion":
    from evolucion_cutoff import render_evolucion_cutoff
    _safe_call(render_evolucion_cutoff)
