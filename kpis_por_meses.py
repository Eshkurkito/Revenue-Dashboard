import pandas as pd
import streamlit as st
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, load_groups

def render_kpis_por_meses(raw):
    with st.sidebar:
        st.header("Selección de grupo de alojamientos")
        groups = load_groups()
        group_names = sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)
        props_rc = groups[selected_group] if selected_group else []

        st.header("Periodo")
        year = st.number_input("Año", min_value=2000, max_value=date.today().year, value=date.today().year)
        start_month = st.number_input("Mes inicial", min_value=1, max_value=12, value=1)
        end_month = st.number_input("Mes final", min_value=1, max_value=12, value=12)

    # Genera rango de meses
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end = pd.Timestamp(year=year, month=end_month, day=1)
    months = pd.date_range(start=start, end=end, freq="MS").to_period("M")

    st.subheader(f"KPI por meses ({year})")

    resultados = []
    for mes in months:
        periodo_inicio = mes.to_timestamp()
        periodo_fin = (mes + 1).to_timestamp() - pd.Timedelta(days=1)
        by_prop, totales = compute_kpis(
            df_all=raw,
            cutoff=periodo_fin,
            period_start=periodo_inicio,
            period_end=periodo_fin,
            filter_props=props_rc
        )
        resultados.append({
            "Mes": mes.strftime("%Y-%m"),
            "Noches ocupadas": totales["noches_ocupadas"],
            "Noches disponibles": totales["noches_disponibles"],
            "Ocupación %": totales["ocupacion_pct"],
            "Ingresos": totales["ingresos"],
            "ADR": totales["adr"],
            "RevPAR": totales["revpar"]
        })

    df_result = pd.DataFrame(resultados)
    st.dataframe(df_result, use_container_width=True)

    # Descarga CSV
    st.download_button(
        "📥 Descargar CSV",
        data=df_result.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"kpis_mensuales_{year}.csv",
        mime="text/csv"
    )