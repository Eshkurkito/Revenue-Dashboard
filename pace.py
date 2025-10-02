import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_pace(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        period_start, period_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "pace"
        )
        props_pace = group_selector(
            "Filtrar alojamientos (opcional)",
            list(raw["Alojamiento"].unique()),
            key_prefix="props_pace",
            default=[]
        )
        inv_pace = st.number_input(
            "Inventario (opcional)",
            min_value=0, value=0, step=1, key="inv_pace"
        )
        d_max = st.slider("Días antes de la estancia (curva D)", min_value=7, max_value=120, value=30, step=1, key="pace_dmax")

    # Curva Pace: para cada D (0..d_max), noches/ingresos confirmados a D días antes de la estancia
    # Aquí deberías tener una función pace_series en utils.py
    from utils import pace_series

    pace_df = pace_series(
        df=raw,
        period_start=pd.to_datetime(period_start),
        period_end=pd.to_datetime(period_end),
        d_max=d_max,
        props=props_pace if props_pace else None,
        inv_override=int(inv_pace) if inv_pace > 0 else None,
    )

    st.subheader("Curva Pace (curva D)")
    help_block("Curva Pace")
    st.line_chart(pace_df.set_index("D")[["noches", "ingresos"]], height=260)
    st.dataframe(pace_df, use_container_width=True)
    csv = pace_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Descargar curva Pace (CSV)",
        data=csv,
        file_name="curva_pace.csv",
        mime="text/csv"
    )