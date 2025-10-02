import streamlit as st
import pandas as pd
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block

def render_resumen_comparativo(raw):
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_rc = st.date_input("Fecha de corte", value=date.today(), key="cutoff_rc")
        c1, c2 = st.columns(2)
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "rc"
        )
        inv_rc = st.number_input(
            "Sobrescribir inventario (nº alojamientos)",
            min_value=0, value=0, step=1, key="inv_rc"
        )
        props_rc = group_selector(
            "Filtrar alojamientos (opcional)",
            list(raw["Alojamiento"].unique()),
            key_prefix="props_rc",
            default=[]
        )
        st.markdown("—")
        compare_rc = st.checkbox(
            "Comparar con año anterior (mismo día/mes)", value=True, key="cmp_rc"
        )
        inv_rc_prev = st.number_input(
            "Inventario año anterior (opcional)",
            min_value=0, value=0, step=1, key="inv_rc_prev"
        )

    # KPIs actuales
    by_prop_rc, total_rc = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_rc),
        period_start=pd.to_datetime(start_rc),
        period_end=pd.to_datetime(end_rc),
        inventory_override=int(inv_rc) if inv_rc > 0 else None,
        filter_props=props_rc if props_rc else None,
    )

    # KPIs año anterior (si compara)
    if compare_rc:
        cutoff_rc_ly = pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1)
        start_rc_ly = pd.to_datetime(start_rc) - pd.DateOffset(years=1)
        end_rc_ly = pd.to_datetime(end_rc) - pd.DateOffset(years=1)
        by_prop_rc_ly, total_rc_ly = compute_kpis(
            df_all=raw,
            cutoff=cutoff_rc_ly,
            period_start=start_rc_ly,
            period_end=end_rc_ly,
            inventory_override=int(inv_rc_prev) if inv_rc_prev > 0 else None,
            filter_props=props_rc if props_rc else None,
        )
    else:
        total_rc_ly = None

    st.subheader("Resumen comparativo")
    help_block("Resumen Comparativo")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_rc['noches_ocupadas']:,}".replace(",", "."),
              delta=f"{(total_rc['noches_ocupadas'] - (total_rc_ly['noches_ocupadas'] if total_rc_ly else 0)):+,.0f}".replace(",", ".") if total_rc_ly else None)
    c2.metric("Noches disponibles", f"{total_rc['noches_disponibles']:,}".replace(",", "."),
              delta=f"{(total_rc['noches_disponibles'] - (total_rc_ly['noches_disponibles'] if total_rc_ly else 0)):+,.0f}".replace(",", ".") if total_rc_ly else None)
    c3.metric("Ocupación", f"{total_rc['ocupacion_pct']:.2f}%",
              delta=f"{(total_rc['ocupacion_pct'] - (total_rc_ly['ocupacion_pct'] if total_rc_ly else 0)):+.2f}%" if total_rc_ly else None)
    c4.metric("Ingresos (€)", f"{total_rc['ingresos']:.2f}",
              delta=f"{(total_rc['ingresos'] - (total_rc_ly['ingresos'] if total_rc_ly else 0)):+.2f}" if total_rc_ly else None)
    c5.metric("ADR (€)", f"{total_rc['adr']:.2f}",
              delta=f"{(total_rc['adr'] - (total_rc_ly['adr'] if total_rc_ly else 0)):+.2f}" if total_rc_ly else None)
    c6.metric("RevPAR (€)", f"{total_rc['revpar']:.2f}",
              delta=f"{(total_rc['revpar'] - (total_rc_ly['revpar'] if total_rc_ly else 0)):+.2f}" if total_rc_ly else None)

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_rc.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_rc, use_container_width=True)
        csv = by_prop_rc.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar detalle (CSV)",
            data=csv,
            file_name="detalle_comparativo_por_alojamiento.csv",
            mime="text/csv"
        )