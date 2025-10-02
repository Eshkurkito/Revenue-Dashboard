import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_cuadro_mando_pro(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_pro = st.date_input("Fecha de corte", value=date.today(), key="cutoff_pro")
        c1, c2 = st.columns(2)
        start_pro, end_pro = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "pro"
        )
        inv_pro = st.number_input(
            "Sobrescribir inventario (nº alojamientos)",
            min_value=0, value=0, step=1, key="inv_pro"
        )
        props_pro = group_selector(
            "Filtrar alojamientos (opcional)",
            list(raw["Alojamiento"].unique()),
            key_prefix="props_pro",
            default=[]
        )

    # KPIs PRO
    by_prop_pro, total_pro = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_pro),
        period_start=pd.to_datetime(start_pro),
        period_end=pd.to_datetime(end_pro),
        inventory_override=int(inv_pro) if inv_pro > 0 else None,
        filter_props=props_pro if props_pro else None,
    )

    st.subheader("Cuadro de mando PRO")
    help_block("Cuadro de mando PRO")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_pro['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_pro['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_pro['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{total_pro['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{total_pro['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{total_pro['revpar']:.2f}")

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_pro.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_pro, use_container_width=True)
        csv = by_prop_pro.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar detalle (CSV)",
            data=csv,
            file_name="detalle_pro_por_alojamiento.csv",
            mime="text/csv"
        )