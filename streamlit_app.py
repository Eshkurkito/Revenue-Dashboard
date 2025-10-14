import pandas as pd
import streamlit as st
from datetime import date
from utils import period_inputs
from landing_ui import render_landing

# --- Define la función aquí ---
def cargar_dataframe():
    # Devuelve un DataFrame vacío por defecto
    return pd.DataFrame()

from consulta_normal import render_consulta_normal
from resumen_comparativo import render_resumen_comparativo
from kpis_por_meses import render_kpis_por_meses
from evolucion_corte import render_evolucion_corte
from pickup import render_pickup
from pace import render_pace
from prediccion_pace import render_prediccion_pace
from cuadro_mando_pro import render_cuadro_mando_pro
from alerts_module import render_alerts_module
from what_if import render_what_if

# Configuración de página
st.set_page_config(page_title="Revenue Dashboard", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 600px at 10% 0%, #0b1220 0%, #0e1117 25%, #111827 60%, #0b0f1a 100%); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.button("⬅️ Volver al inicio"):
        st.session_state.view = "landing"
        st.experimental_rerun()

if "view" not in st.session_state:
    st.session_state.view = "landing"

if st.session_state.view == "landing":
    render_landing()

elif st.session_state.view == "consulta":
    from consulta_normal import render_consulta_normal
    render_consulta_normal()

elif st.session_state.view == "pro":
    from cuadro_mando_pro import render_cuadro_mando_pro
    render_cuadro_mando_pro()

elif st.session_state.view == "whatif":
    from what_if import render_what_if
    raw = st.session_state.get("df_active") or st.session_state.get("raw")
    try:
        render_what_if(raw)
    except TypeError:
        render_what_if()

# -------- Sidebar: carga de datos --------
st.sidebar.header("Carga de datos")

# Subida de archivo
uploaded_file = st.sidebar.file_uploader("Carga tu archivo Excel/CSV", type=["xlsx", "csv"])

# Botón para confirmar el archivo
if uploaded_file is not None:
    if st.sidebar.button("Usar este archivo"):
        if uploaded_file.name.endswith(".xlsx"):
            st.session_state.raw = pd.read_excel(uploaded_file)
        else:
            st.session_state.raw = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente.")
else:
    if "raw" not in st.session_state:
        st.session_state.raw = cargar_dataframe()  # Reemplaza por tu función de carga
raw = st.session_state.raw

# Menú de modos
st.sidebar.header("Menú principal")
mode = st.sidebar.selectbox(
    "Selecciona modo",
    [
        "Consulta normal",
        "Resumen Comparativo",
        "KPIs por meses",
        "Evolución por fecha de corte",
        "Pickup (entre dos cortes)",
        "Pace (curva D)",
        "Predicción (Pace)",
        "Cuadro de mando (PRO)",
        "Panel de alertas",
        "What‑if (escenarios)"
    ]
)

# Routing: llama al modo seleccionado
if raw is None:
    st.warning("⚠️ No hay datos cargados. Sube tu archivo en la barra lateral.")
else:
    if mode == "Consulta normal":
        render_consulta_normal(raw)
    elif mode == "Resumen Comparativo":
        render_resumen_comparativo(raw)
    elif mode == "KPIs por meses":
        render_kpis_por_meses(raw)
    elif mode == "Evolución por fecha de corte":
        render_evolucion_corte(raw)
    elif mode == "Pickup (entre dos cortes)":
        render_pickup(raw)
    elif mode == "Pace (curva D)":
        render_pace(raw)
    elif mode == "Predicción (Pace)":
        render_prediccion_pace(raw)
    elif mode == "Cuadro de mando (PRO)":
        render_cuadro_mando_pro(raw)
    elif mode == "Panel de alertas":
        render_alerts_module(raw)
    elif mode == "What‑if (escenarios)":
        render_what_if(raw)
