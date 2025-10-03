import pandas as pd
import streamlit as st
from datetime import date
from utils import compute_kpis, period_inputs, group_selector, help_block, save_group_csv, load_groups, GROUPS_PATH

def render_resumen_comparativo(raw):
    if raw is None:
        st.stop()
    raw.columns = [col.strip() for col in raw.columns]

    with st.sidebar:
        st.header("Parámetros")
        cutoff_rc = st.date_input("Fecha de corte", value=date.today(), key="cutoff_rc")
        fecha_fin_mes = (pd.Timestamp.today() + pd.offsets.MonthEnd(0)).date()
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            fecha_fin_mes,
            "rc_period"
        )

        st.header("Gestión de grupos")
        groups = load_groups()
        group_names = ["Ninguno"] + sorted(list(groups.keys()))
        selected_group = st.selectbox("Grupo guardado", group_names)

        if selected_group and selected_group != "Ninguno":
            props_rc = groups[selected_group]
            if st.button(f"Eliminar grupo '{selected_group}'"):
                df = pd.read_csv(GROUPS_PATH)
                df = df[df["Grupo"] != selected_group]
                df.to_csv(GROUPS_PATH, index=False)
                st.success(f"Grupo '{selected_group}' eliminado.")
                st.experimental_rerun()
        else:
            if "Alojamiento" not in raw.columns:
                st.warning("No se encontró la columna 'Alojamiento'. Sube un archivo válido o revisa el nombre de la columna.")
                st.stop()
            props_rc = group_selector(
                "Filtrar alojamientos (opcional)",
                sorted([str(x) for x in raw["Alojamiento"].dropna().unique()]),
                key_prefix="props_rc",
                default=[]
            )

        group_name = st.text_input("Nombre del grupo para guardar")
        if st.button("Guardar grupo de pisos") and group_name and props_rc:
            save_group_csv(group_name, props_rc)
            st.success(f"Grupo '{group_name}' guardado.")

        compare_rc = st.checkbox(
            "Comparar con año anterior (mismo día/mes)", value=True, key="cmp_rc"
        )

    # KPIs actuales
    by_prop_rc, total_rc = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_rc),
        period_start=pd.to_datetime(start_rc),
        period_end=pd.to_datetime(end_rc),
        filter_props=props_rc if props_rc else None,
    )

    # KPIs año anterior
    if compare_rc:
        ly_start = pd.to_datetime(start_rc) - pd.DateOffset(years=1)
        ly_end = pd.to_datetime(end_rc) - pd.DateOffset(years=1)
        ly_cutoff = pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1)
        by_prop_ly, total_ly = compute_kpis(
            df_all=raw,
            cutoff=ly_cutoff,
            period_start=ly_start,
            period_end=ly_end,
            filter_props=props_rc if props_rc else None,
        )
    else:
        by_prop_ly = pd.DataFrame()

    # Unir ambos detalles por alojamiento
    detalle = by_prop_rc.copy()
    if not by_prop_ly.empty:
        detalle = detalle.merge(
            by_prop_ly[["Alojamiento", "noches_ocupadas", "ocupacion_pct", "adr", "ingresos", "revpar"]],
            on="Alojamiento", how="left", suffixes=('', '_ly')
        )
        # Renombrar columnas para claridad
        detalle.rename(columns={
            "noches_ocupadas": "Noches ocupadas",
            "noches_ocupadas_ly": "Noches ocupadas LY",
            "ocupacion_pct": "Ocupación",
            "ocupacion_pct_ly": "Ocupación LY",
            "adr": "ADR",
            "adr_ly": "ADR LY",
            "ingresos": "Ingresos",
            "ingresos_ly": "Ingresos LY",
            "revpar_ly": "Ingresos finales LY"
        }, inplace=True)
        # Si la columna 'revpar_ly' no existe, crea vacía
        if "Ingresos finales LY" not in detalle.columns:
            detalle["Ingresos finales LY"] = None
    else:
        detalle.rename(columns={
            "noches_ocupadas": "Noches ocupadas",
            "ocupacion_pct": "Ocupación",
            "adr": "ADR",
            "ingresos": "Ingresos"
        }, inplace=True)
        detalle["Noches ocupadas LY"] = None
        detalle["Ocupación LY"] = None
        detalle["ADR LY"] = None
        detalle["Ingresos LY"] = None
        detalle["Ingresos finales LY"] = None

    # Formato y colores
    def color_diff(val, ly_val):
        if pd.isnull(val) or pd.isnull(ly_val):
            return ""
        return "background-color: #d4f7d4" if val >= ly_val else "background-color: #ffd6d6"

    def euro_fmt(val):
        if pd.isnull(val):
            return ""
        return f"{val:,.2f} €"

    def pct_fmt(val):
        if pd.isnull(val):
            return ""
        return f"{val:.2f}%"

    # Aplica formato y colores con Styler
    cols_num = ["Noches ocupadas", "Noches ocupadas LY", "Ocupación", "Ocupación LY", "ADR", "ADR LY", "Ingresos", "Ingresos LY", "Ingresos finales LY"]
    detalle_styler = detalle.style.format({
        "ADR": euro_fmt,
        "ADR LY": euro_fmt,
        "Ingresos": euro_fmt,
        "Ingresos LY": euro_fmt,
        "Ingresos finales LY": euro_fmt,
        "Ocupación": pct_fmt,
        "Ocupación LY": pct_fmt,
        "Noches ocupadas": "{:.0f}",
        "Noches ocupadas LY": "{:.0f}",
    })

    # Colores verde/rojo según mejora o empeora respecto LY
    for col, ly_col in [
        ("Noches ocupadas", "Noches ocupadas LY"),
        ("Ocupación", "Ocupación LY"),
        ("ADR", "ADR LY"),
        ("Ingresos", "Ingresos LY"),
        ("Ingresos finales LY", "Ingresos finales LY")
    ]:
        if col in detalle.columns and ly_col in detalle.columns:
            detalle_styler = detalle_styler.apply(
                lambda x: [color_diff(x[col], x[ly_col]) for i in x.index], axis=1, subset=[col]
            )

    st.subheader("Detalle por alojamiento")
    st.dataframe(detalle_styler, use_container_width=True)

    # Exportar a Excel con formato
    import io
    from pandas.io.formats.style import Styler

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        detalle_styler.to_excel(writer, sheet_name="Detalle", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Detalle"]
        # Ajusta el ancho de columnas y formato numérico
        for idx, col in enumerate(detalle.columns):
            worksheet.set_column(idx, idx, 18)
        # Los colores ya se exportan con Styler

    st.download_button(
        "📥 Descargar detalle por alojamiento (Excel)",
        data=output.getvalue(),
        file_name="detalle_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )