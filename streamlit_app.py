import pandas as pd
import streamlit as st
from landing_ui import render_landing, get_logo_path, LOGO_PATH  # ← añade get_logo_path

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
st.markdown("""
<style>
:root{ --brand:#2e485f; --brand-600:#264052; --brand-50:#f6f8fb; }
.stApp { background:#ffffff; }
.block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] { background: var(--brand-50); }
.stButton > button, .btn-primary { background: var(--brand); color: #fff; border: 1px solid var(--brand-600); border-radius: 10px; }
.stButton > button:hover, .btn-primary:hover { background: var(--brand-600); border-color: var(--brand-600); }
[data-testid="stMetricDelta"] { color: var(--brand) !important; }
[data-testid="stTable"], .stDataFrame thead tr th { background: var(--brand-50) !important; }
a, .st-af { color: var(--brand); }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: logo + carga de datos ---
with st.sidebar:
    logo = get_logo_path()
    if logo:
        st.image(logo, use_container_width=True)  # antes: use_column_width=True
    else:
        st.write("Florit Flats")
    if st.button("⬅️ Volver al inicio"):
        st.session_state.view = "landing"
        _rerun()

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
