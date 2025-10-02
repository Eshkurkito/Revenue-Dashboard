import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block, compute_portal_share

def render_consulta_normal(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_normal = st.date_input("Fecha de corte", value=date.today(), key="cutoff_normal")
        c1, c2 = st.columns(2)
        start_normal, end_normal = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "normal"
        )
        inv_normal = st.number_input(
            "Sobrescribir inventario (nº alojamientos)",
            min_value=0, value=0, step=1, key="inv_normal"
        )
        props_normal = group_selector(
            "Filtrar alojamientos (opcional)",
            list(raw["Alojamiento"].unique()),
            key_prefix="props_normal",
            default=[]
        )
        st.markdown("—")
        compare_normal = st.checkbox(
            "Comparar con año anterior (mismo día/mes)", value=False, key="cmp_normal"
        )
        inv_normal_prev = st.number_input(
            "Inventario año anterior (opcional)",
            min_value=0, value=0, step=1, key="inv_normal_prev"
        )

    # Cálculo base
    by_prop_n, total_n = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        inventory_override=int(inv_normal) if inv_normal > 0 else None,
        filter_props=props_normal if props_normal else None,
    )

    st.subheader("Resultados totales")
    help_block("Consulta normal")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_n['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_n['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_n['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{total_n['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{total_n['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{total_n['revpar']:.2f}")

    # Distribución por portal (si existe columna)
    port_df = compute_portal_share(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        filter_props=props_normal if props_normal else None,
    )
    st.subheader("Distribución por portal (reservas en el periodo)")
    if port_df is None:
        st.info("No se encontró la columna 'Portal'. Si tiene otro nombre, dímelo y lo mapeo.")
    elif port_df.empty:
        st.warning("No hay reservas del periodo a la fecha de corte para calcular distribución por portal.")
    else:
        port_view = port_df.copy()
        port_view["% Reservas"] = port_view["% Reservas"].round(2)
        st.dataframe(port_view, use_container_width=True)
        csv_port = port_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar distribución por portal (CSV)",
            data=csv_port,
            file_name="portales_distribucion.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_n.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_n, use_container_width=True)
        csv = by_prop_n.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar detalle (CSV)",
            data=csv,
            file_name="detalle_por_alojamiento.csv",
            mime="text/csv"
        )