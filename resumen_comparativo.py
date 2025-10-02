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
        # Unir por alojamiento y calcular diferencias
        df_comp = by_prop_rc.merge(
            by_prop_rc_ly,
            on="Alojamiento",
            suffixes=("", "_ly"),
            how="left"
        )
        # Añadir columnas de diferencia, comprobando existencia y usando fillna(0)
        for col in ["noches_ocupadas", "noches_disponibles", "ocupacion_pct", "ingresos", "adr", "revpar"]:
            col_ly = f"{col}_ly"
            if col in df_comp.columns and col_ly in df_comp.columns:
                df_comp[f"diff_{col}"] = df_comp[col].fillna(0) - df_comp[col_ly].fillna(0)
            elif col in df_comp.columns:
                df_comp[f"diff_{col}"] = df_comp[col].fillna(0)
            else:
                df_comp[f"diff_{col}"] = 0
        # Formato condicional para colores
        def color_diff(val):
            if pd.isnull(val):
                return ""
            return f"background-color: {'#b6fcb6' if val > 0 else '#ffb6b6'}"  # verde si >0, rojo si <=0
        styled_df = df_comp.style.applymap(color_diff, subset=[f"diff_{col}" for col in ["noches_ocupadas", "noches_disponibles", "ocupacion_pct", "ingresos", "adr", "revpar"]])
    else:
        total_rc_ly = None
        df_comp = by_prop_rc.copy()
        styled_df = df_comp.style

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
    if df_comp.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.write("Las columnas 'diff_' muestran la diferencia respecto al año anterior. Verde = superior, rojo = inferior.")
        st.dataframe(styled_df, use_container_width=True)
        # Exportar a Excel
        import io
        output = io.BytesIO()
        df_comp.to_excel(output, index=False, sheet_name="Comparativo")
        output.seek(0)
        st.download_button(
            "📥 Descargar detalle (Excel)",
            data=output,
            file_name="detalle_comparativo_por_alojamiento.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )