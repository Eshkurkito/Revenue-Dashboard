import pandas as pd
import streamlit as st
from datetime import date
from utils import period_inputs, group_selector, save_group_csv, load_groups, GROUPS_PATH

def calcular_kpis_por_alojamiento(df, fecha_inicio, fecha_fin):
    # Calcula noches ocupadas
    df["Noches ocupadas"] = (
        pd.to_datetime(df["Fecha salida"]) - pd.to_datetime(df["Fecha entrada"])
    ).dt.days

    # Días en el periodo
    dias_periodo = (fecha_fin - fecha_inicio).days + 1
    alojamientos = df["Alojamiento"].unique()
    noches_posibles = dias_periodo * len(alojamientos)

    agrupado = df.groupby("Alojamiento").agg(
        noches_ocupadas=("Noches ocupadas", "sum"),
        ingresos=("Alquiler con IVA (€)", "sum"),
        reservas=("Alojamiento", "count")
    ).reset_index()

    agrupado["ADR"] = agrupado["ingresos"] / agrupado["noches_ocupadas"]
    agrupado["Ocupación"] = agrupado["noches_ocupadas"] / dias_periodo * 100

    return agrupado

def euro_fmt(val):
    if pd.isnull(val):
        return ""
    return f"{val:,.2f} €"

def pct_fmt(val):
    if pd.isnull(val):
        return ""
    return f"{val:.2f}%"

def color_diff(val, ly_val):
    if pd.isnull(val) or pd.isnull(ly_val):
        return ""
    return "background-color: #d4f7d4" if val >= ly_val else "background-color: #ffd6d6"

def color_row(row, col, ly_col):
    try:
        return [color_diff(row[col], row[ly_col])]
    except Exception:
        return [""]

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

    # Filtra por fechas del periodo actual
    df_actual = raw[
        (pd.to_datetime(raw["Fecha entrada"]) >= pd.to_datetime(start_rc)) &
        (pd.to_datetime(raw["Fecha entrada"]) <= pd.to_datetime(end_rc))
    ]
    if props_rc:
        df_actual = df_actual[df_actual["Alojamiento"].isin(props_rc)]

    detalle_actual = calcular_kpis_por_alojamiento(df_actual, pd.to_datetime(start_rc), pd.to_datetime(end_rc))

    # Si comparar con LY
    if compare_rc:
        ly_start = pd.to_datetime(start_rc) - pd.DateOffset(years=1)
        ly_end = pd.to_datetime(end_rc) - pd.DateOffset(years=1)
        df_ly = raw[
            (pd.to_datetime(raw["Fecha entrada"]) >= ly_start) &
            (pd.to_datetime(raw["Fecha entrada"]) <= ly_end)
        ]
        if props_rc:
            df_ly = df_ly[df_ly["Alojamiento"].isin(props_rc)]
        detalle_ly = calcular_kpis_por_alojamiento(df_ly, ly_start, ly_end)
        # Merge ambos detalles
        detalle = detalle_actual.merge(
            detalle_ly[["Alojamiento", "noches_ocupadas", "Ocupación", "ADR", "ingresos"]],
            on="Alojamiento", how="left", suffixes=('', '_LY')
        )
        # Renombra para claridad
        detalle.rename(columns={
            "noches_ocupadas": "Noches ocupadas",
            "noches_ocupadas_LY": "Noches ocupadas LY",
            "Ocupación": "Ocupación",
            "Ocupación_LY": "Ocupación LY",
            "ADR": "ADR",
            "ADR_LY": "ADR LY",
            "ingresos": "Ingresos",
            "ingresos_LY": "Ingresos LY"
        }, inplace=True)
    else:
        detalle = detalle_actual.copy()
        detalle["Noches ocupadas LY"] = None
        detalle["Ocupación LY"] = None
        detalle["ADR LY"] = None
        detalle["Ingresos LY"] = None

    # Añade columna de ingresos finales LY (puedes adaptar la lógica si tienes otra fuente)
    detalle["Ingresos finales LY"] = detalle["Ingresos LY"]

    # Ordena columnas intercaladas
    columnas_finales = [
        "Alojamiento",
        "Noches ocupadas", "Noches ocupadas LY",
        "Ocupación", "Ocupación LY",
        "ADR", "ADR LY",
        "Ingresos", "Ingresos LY",
        "Ingresos finales LY"
    ]
    detalle = detalle[columnas_finales]

    # Formato y colores con Styler
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

    for col, ly_col in [
        ("Noches ocupadas", "Noches ocupadas LY"),
        ("Ocupación", "Ocupación LY"),
        ("ADR", "ADR LY"),
        ("Ingresos", "Ingresos LY"),
    ]:
        if col in detalle.columns and ly_col in detalle.columns:
            detalle_styler = detalle_styler.apply(
                lambda s: [
                    color_diff(val, ly_val)
                    for val, ly_val in zip(s, detalle[ly_col])
                ],
                subset=[col]
            )

    st.subheader("Detalle por alojamiento")
    st.dataframe(detalle_styler, use_container_width=True)

    # Exportar a Excel con formato
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        detalle.to_excel(writer, sheet_name="Detalle", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Detalle"]
        for idx, col in enumerate(detalle.columns):
            worksheet.set_column(idx, idx, 18)
    st.download_button(
        "📥 Descargar detalle por alojamiento (Excel)",
        data=output.getvalue(),
        file_name="detalle_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )