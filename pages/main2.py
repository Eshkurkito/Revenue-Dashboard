import streamlit as st
import pandas as pd
from pathlib import Path

from utils import parse_dates, load_groups

st.set_page_config(page_title="Revenue PRO – Portada", layout="wide")

@st.cache_data(show_spinner=False)
def _parse_dates_cached(df: pd.DataFrame) -> pd.DataFrame:
    return parse_dates(df)

def load_data() -> pd.DataFrame:
    # Reusar si ya está en memoria
    raw = st.session_state.get("raw")
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        return raw

    up = st.sidebar.file_uploader("Sube reservas (CSV o Excel)", type=["csv", "xlsx", "xls"])
    if not up:
        return pd.DataFrame()

    ext = Path(up.name).suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(up)
        else:
            # Lee la primera hoja por defecto (evita reruns por selector de hoja)
            df = pd.read_excel(up, sheet_name=0)
        df = _parse_dates_cached(df)
        st.session_state["raw"] = df
        return df
    except Exception as e:
        st.error(f"No se pudo leer el archivo ({ext}). Detalle: {e}")
        return pd.DataFrame()

# ---------------- Sidebar: grupos (solo lectura) ----------------
st.sidebar.subheader("Grupos de propiedades")
groups = load_groups()
if not groups:
    st.sidebar.info("No hay grupos guardados (grupos_guardados.csv).")

group_names = ["Todos"] + sorted(groups.keys())
selected_group = st.sidebar.selectbox("Grupo activo", group_names, key="global_group")

props = [] if selected_group == "Todos" else groups.get(selected_group, [])
st.session_state["active_group_name"] = None if selected_group == "Todos" else selected_group
st.session_state["active_props"] = props

with st.sidebar.expander("Alojamientos del grupo", expanded=False):
    if props:
        st.dataframe(pd.DataFrame({"Alojamiento": props}), use_container_width=True, hide_index=True)
    else:
        st.caption("Todos los alojamientos")

# ---------------- Portada ----------------
st.title("Revenue PRO – Portada")

raw = load_data()
if raw.empty:
    st.info("Sube un CSV/Excel de reservas en la barra lateral para empezar.")
    st.stop()

# Filtrado por grupo
if props:
    df_view = raw[raw["Alojamiento"].astype(str).isin(props)].copy()
else:
    df_view = raw.copy()

# Publicar DF filtrado para el resto de páginas
st.session_state["df_active"] = df_view

st.caption(f"Grupo: {selected_group} · Alojamientos activos: {df_view['Alojamiento'].nunique():,}".replace(",", "."))
c1, c2, c3 = st.columns(3)
c1.metric("Reservas", f"{len(df_view):,}".replace(",", "."))
c2.metric("Alojamientos", f"{df_view['Alojamiento'].nunique():,}".replace(",", "."))
total_ing = pd.to_numeric(df_view.get("Alquiler con IVA (€)", 0), errors="coerce").fillna(0).sum()
c3.metric("Ingresos (total)", f"{total_ing:,.2f}".replace(",", "."))

st.dataframe(df_view.head(50), use_container_width=True)