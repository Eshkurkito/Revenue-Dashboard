import streamlit as st
import pandas as pd
from pathlib import Path
from utils import parse_dates, load_groups  # ← solo lectura de grupos

# Sidebar: grupos (solo lectura desde CSV)
st.sidebar.subheader("Grupos de propiedades")
groups = load_groups()
if not groups:
    st.sidebar.info("No hay grupos guardados (grupos_guardados.csv).")

group_names = ["Todos"] + sorted(groups.keys())
selected_group = st.sidebar.selectbox("Grupo activo", group_names, key="global_group")

props = [] if selected_group == "Todos" else groups.get(selected_group, [])
st.session_state["active_props"] = props
st.sidebar.caption(f"Propiedades activas: {len(props) if props else 'todas'}")

# Resto del código...
st.write(f"Propiedades activas: {len(props) if props else 'todas'}")