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
        # Asegurar columnas necesarias existen
        for col in ["noches_ocupadas", "noches_disponibles", "ingresos", "adr"]:
            if col not in df_comp.columns:
                df_comp[col] = 0
            if f"{col}_ly" not in df_comp.columns:
                df_comp[f"{col}_ly"] = 0
        # Nueva columna de ocupación actual y año anterior
        df_comp["ocupacion"] = df_comp["noches_ocupadas"] / df_comp["noches_disponibles"].replace(0, pd.NA)
        df_comp["ocupacion_ly"] = df_comp["noches_ocupadas_ly"] / df_comp["noches_disponibles_ly"].replace(0, pd.NA)
        # Añadir columnas de diferencia solo para noches ocupadas, ingresos y adr
        for col in ["noches_ocupadas", "ingresos", "adr"]:
            col_ly = f"{col}_ly"
            df_comp[f"diff_{col}"] = df_comp[col].fillna(0) - df_comp[col_ly].fillna(0)
        # Diferencia de ocupación
        df_comp["diff_ocupacion"] = (df_comp["ocupacion"] - df_comp["ocupacion_ly"]).fillna(0)
        # Ingresos finales: recalcular usando compute_kpis sin cutoff
        by_prop_final, _ = compute_kpis(
            df_all=raw,
            cutoff=pd.Timestamp.max,
            period_start=pd.to_datetime(start_rc),
            period_end=pd.to_datetime(end_rc),
            inventory_override=int(inv_rc) if inv_rc > 0 else None,
            filter_props=props_rc if props_rc else None,
        )
        # Asegurar columna 'ingresos' existe en by_prop_final
        if "ingresos" not in by_prop_final.columns:
            by_prop_final["ingresos"] = 0
        df_comp = df_comp.merge(
            by_prop_final[["Alojamiento", "ingresos"]].rename(columns={"ingresos": "ingresos_finales"}),
            on="Alojamiento",
            how="left"
        )
        # Asegurar columnas 'ingresos' e 'ingresos_finales' existen en df_comp
        if "ingresos" not in df_comp.columns:
            df_comp["ingresos"] = 0
        if "ingresos_finales" not in df_comp.columns:
            df_comp["ingresos_finales"] = 0
        df_comp["diff_ingresos_finales"] = (df_comp["ingresos_finales"] - df_comp["ingresos"]).fillna(0)
        # Formato condicional solo para las diferencias relevantes
        def color_diff(val):
            if pd.isnull(val):
                return ""
            return f"background-color: {'#b6fcb6' if val > 0 else '#ffb6b6'}"
        styled_df = df_comp.style.applymap(color_diff, subset=[
            "diff_noches_ocupadas", "diff_ingresos", "diff_adr", "diff_ocupacion", "diff_ingresos_finales"
        ])
    else:
        total_rc_ly = None
        df_comp = by_prop_rc.copy()
        # Asegurar columnas necesarias existen
        for col in ["noches_ocupadas", "noches_disponibles", "ingresos"]:
            if col not in df_comp.columns:
                df_comp[col] = 0
        # Nueva columna de ocupación
        df_comp["ocupacion"] = df_comp["noches_ocupadas"] / df_comp["noches_disponibles"].replace(0, pd.NA)
        # Ingresos finales
        by_prop_final, _ = compute_kpis(
            df_all=raw,
            cutoff=pd.Timestamp.max,
            period_start=pd.to_datetime(start_rc),
            period_end=pd.to_datetime(end_rc),
            inventory_override=int(inv_rc) if inv_rc > 0 else None,
            filter_props=props_rc if props_rc else None,
        )
        if "ingresos" not in by_prop_final.columns:
            by_prop_final["ingresos"] = 0
        df_comp = df_comp.merge(
            by_prop_final[["Alojamiento", "ingresos"]].rename(columns={"ingresos": "ingresos_finales"}),
            on="Alojamiento",
            how="left"
        )
        if "ingresos" not in df_comp.columns:
            df_comp["ingresos"] = 0
        if "ingresos_finales" not in df_comp.columns:
            df_comp["ingresos_finales"] = 0
        df_comp["diff_ingresos_finales"] = (df_comp["ingresos_finales"] - df_comp["ingresos"]).fillna(0)
        def color_diff(val):
            if pd.isnull(val):
                return ""
            return f"background-color: {'#b6fcb6' if val > 0 else '#ffb6b6'}"
        styled_df = df_comp.style.applymap(color_diff, subset=["diff_ingresos_finales"])

    # Al final del bloque if/else, justo antes de mostrar la tabla:
    # Selecciona solo las columnas relevantes y ordénalas
    cols_to_show = [
        "Alojamiento",
        "noches_ocupadas", "ingresos", "adr",
        "noches_ocupadas_ly", "ingresos_ly", "adr_ly",
        "ocupacion", "ocupacion_ly",
        "diff_noches_ocupadas", "diff_ingresos", "diff_adr", "diff_ocupacion",
        "ingresos_finales", "diff_ingresos_finales"
    ]
    # Filtra las columnas que realmente existen en el DataFrame
    cols_to_show = [col for col in cols_to_show if col in df_comp.columns]
    df_comp = df_comp[cols_to_show].copy()

    # Formatea ocupación como porcentaje con dos decimales, evitando None
    for col in ["ocupacion", "ocupacion_ly"]:
        if col in df_comp.columns:
            df_comp[col] = (df_comp[col].fillna(0) * 100).round(2).astype(str) + "%"

    # Formato condicional solo en las columnas de diferencia
    diff_cols = [col for col in df_comp.columns if col.startswith("diff_")]
    def color_diff(val):
        try:
            v = float(str(val).replace("%", ""))
        except:
            return ""
        if pd.isnull(v):
            return ""
        return f"background-color: {'#b6fcb6' if v > 0 else '#ffb6b6'}"

    styled_df = df_comp.style.applymap(color_diff, subset=diff_cols)

    st.subheader("Resumen comparativo")
    help_block("Resumen Comparativo")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_rc.get('noches_ocupadas',0):,}".replace(",", "."),
              delta=f"{(total_rc.get('noches_ocupadas',0) - (total_rc_ly.get('noches_ocupadas',0) if total_rc_ly else 0)):+,.0f}".replace(",", ".") if total_rc_ly else None)
    c2.metric("Noches disponibles", f"{total_rc.get('noches_disponibles',0):,}".replace(",", "."),
              delta=f"{(total_rc.get('noches_disponibles',0) - (total_rc_ly.get('noches_disponibles',0) if total_rc_ly else 0)):+,.0f}".replace(",", ".") if total_rc_ly else None)
    c3.metric("Ocupación", f"{total_rc.get('ocupacion_pct',0):.2f}%",
              delta=f"{(total_rc.get('ocupacion_pct',0) - (total_rc_ly.get('ocupacion_pct',0) if total_rc_ly else 0)):+.2f}%" if total_rc_ly else None)
    c4.metric("Ingresos (€)", f"{total_rc.get('ingresos',0):.2f}",
              delta=f"{(total_rc.get('ingresos',0) - (total_rc_ly.get('ingresos',0) if total_rc_ly else 0)):+.2f}" if total_rc_ly else None)
    c5.metric("ADR (€)", f"{total_rc.get('adr',0):.2f}",
              delta=f"{(total_rc.get('adr',0) - (total_rc_ly.get('adr',0) if total_rc_ly else 0)):+.2f}" if total_rc_ly else None)
    c6.metric("RevPAR (€)", f"{total_rc.get('revpar',0):.2f}",
              delta=f"{(total_rc.get('revpar',0) - (total_rc_ly.get('revpar',0) if total_rc_ly else 0)):+.2f}" if total_rc_ly else None)

    st.divider()
    st.subheader("Detalle por alojamiento")
    if df_comp.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.write(
            "Las columnas 'diff_' muestran la diferencia respecto al año anterior. "
            "Verde = superior, rojo = inferior. "
            "La columna 'ocupacion' es noches ocupadas / noches disponibles. "
            "La columna 'diff_ingresos_finales' compara ingresos a fecha de corte vs ingresos finales."
        )
        st.dataframe(styled_df, use_container_width=True)
        # Exportar a Excel solo las columnas relevantes
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