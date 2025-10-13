import streamlit as st
import pandas as pd
from pathlib import Path
from utils import parse_dates, load_groups  # ← solo lectura

# Sidebar: grupos (solo lectura desde CSV)
st.sidebar.subheader("Grupos de propiedades")
groups = load_groups()
if not groups:
    st.sidebar.info("No hay grupos guardados (grupos_guardados.csv).")

group_names = ["Todos"] + sorted(groups.keys())
selected_group = st.sidebar.selectbox("Grupo activo", group_names, key="global_group")

props = [] if selected_group == "Todos" else groups.get(selected_group, [])
st.session_state["active_group_name"] = None if selected_group == "Todos" else selected_group
st.session_state["active_props"] = props

# Lista de alojamientos del grupo (sidebar)
with st.sidebar.expander("Alojamientos del grupo", expanded=False):
    if props:
        import pandas as pd as _pd  # evitar conflictos de nombres
        st.dataframe(_pd.DataFrame({"Alojamiento": props}), use_container_width=True, hide_index=True)
    else:
        st.caption("Todos los alojamientos")

st.title("Revenue PRO – Portada")
raw = st.file_uploader("Sube un CSV/Excel de reservas", type=["csv", "xlsx"], key="raw_data")

if raw.empty:
    st.info("Sube un CSV/Excel de reservas en la barra lateral para empezar.")
else:
    # Filtro por grupo
    if props:
        df_view = raw[raw["Alojamiento"].astype(str).isin(props)].copy()
    else:
        df_view = raw.copy()

    # Publica el DF filtrado para las páginas
    st.session_state["df_active"] = df_view

    st.caption(f"Grupo: {selected_group} · Alojamientos activos: {df_view['Alojamiento'].nunique():,}".replace(",", "."))
    c1, c2, c3 = st.columns(3)
    c1.metric("Reservas", f"{len(df_view):,}".replace(",", "."))
    c2.metric("Alojamientos", f"{df_view['Alojamiento'].nunique():,}".replace(",", "."))
    c3.metric("Ingresos (total)", f"{pd.to_numeric(df_view.get('Alquiler con IVA (€)', 0)).sum():,.2f}".replace(",", "."))

    st.dataframe(df_view.head(30), use_container_width=True)