import pandas as pd
import streamlit as st
from landing_ui import render_landing, get_logo_path, LOGO_PATH  # ← añade get_logo_path
from auth import require_login, logout_button

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

def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# --- página y tema ---
st.set_page_config(page_title="Revenue Dashboard", page_icon="📊", layout="wide")
st.markdown("""<style>/* tu CSS */</style>""", unsafe_allow_html=True)

from auth import require_login
if not require_login():
    st.stop()

# --- Sidebar: logo + carga de datos ---
with st.sidebar:
    logo = get_logo_path()
    if logo:
        st.image(logo, width=180)  # ajusta 140–200 según prefieras
    else:
        st.write("Florit Flats")
    if st.button("⬅️ Volver al inicio", key="btn_home", use_container_width=True):
        st.session_state.view = "landing"
        _rerun()

    logout_button()  # ← aquí debajo
    st.header("Carga de datos")
    uploaded_file = st.file_uploader("Carga tu archivo Excel/CSV", type=["xlsx", "csv"])
    if uploaded_file is not None and st.button("Usar este archivo"):
        if uploaded_file.name.endswith(".xlsx"):
            st.session_state.raw = pd.read_excel(uploaded_file)
        else:
            st.session_state.raw = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente.")
    else:
        if "raw" not in st.session_state:
            st.session_state.raw = cargar_dataframe()

# --- Router único (sin menú clásico duplicado) ---
if "view" not in st.session_state:
    st.session_state.view = "landing"

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
