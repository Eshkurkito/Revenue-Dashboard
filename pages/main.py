import streamlit as st
import pandas as pd
from pathlib import Path
from utils import parse_dates, load_groups, save_group_csv, delete_group_csv

st.set_page_config(page_title="Revenue PRO", layout="wide")

@st.cache_data(show_spinner=False)
def _parse_dates_cached(df: pd.DataFrame) -> pd.DataFrame:
    return parse_dates(df)

def load_data() -> pd.DataFrame:
    # Si ya está en memoria, úsalo
    if isinstance(st.session_state.get("raw"), pd.DataFrame) and not st.session_state["raw"].empty:
        return st.session_state["raw"]

    up = st.sidebar.file_uploader("Sube reservas (CSV o Excel)", type=["csv", "xlsx", "xls"])
    if not up:
        return pd.DataFrame()

    ext = Path(up.name).suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(up)
        else:
            # Excel: permite elegir hoja
            xls = pd.ExcelFile(up)
            sheet = st.sidebar.selectbox("Hoja Excel", xls.sheet_names, index=0, key="xlsx_sheet")
            df = xls.parse(sheet)
        df = _parse_dates_cached(df)
        st.session_state["raw"] = df
        return df
    except Exception as e:
        st.error(f"No se pudo leer el archivo ({ext}). Revisa el formato. Detalle: {e}")
        return pd.DataFrame()

st.sidebar.title("Parámetros globales")
raw = load_data()

# Gestión de grupos (global)
st.sidebar.subheader("Grupos de propiedades")
groups = load_groups()
group_names = ["Ninguno"] + sorted(groups.keys())
selected_group = st.sidebar.selectbox("Grupo activo", group_names, key="global_group")

if selected_group != "Ninguno":
    props = groups[selected_group]
else:
    props = []

# Crear/editar/eliminar grupos desde el dataset
if not raw.empty:
    st.sidebar.caption("Crear/editar grupos:")
    all_props = sorted(raw["Alojamiento"].dropna().astype(str).unique())
    sel_props = st.sidebar.multiselect("Alojamientos", options=all_props, key="grp_edit_props")
    new_name = st.sidebar.text_input("Nombre del grupo", key="grp_new_name")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Guardar grupo"):
        if new_name and sel_props:
            save_group_csv(new_name, sel_props)
            st.sidebar.success("Grupo guardado.")
            st.experimental_rerun()
    if c2.button("Eliminar grupo activo") and selected_group != "Ninguno":
        delete_group_csv(selected_group)
        st.sidebar.success("Grupo eliminado.")
        st.experimental_rerun()

# Publica la selección global en session_state para que la lean las páginas
st.session_state["active_props"] = props

st.title("Revenue PRO – Portada")
if raw.empty:
    st.info("Sube un CSV de reservas en la barra lateral para empezar.")
else:
    st.caption(f"Propiedades activas: {len(props) if props else 'todas'}")
    st.dataframe(raw.head(30), use_container_width=True)
    st.write("Navega a las páginas del menú (arriba a la izquierda) para ver Overview y Pace/Forecast.")